from collections import namedtuple
import functools

import chex
import haiku as hk
import jax
from ml_collections import config_dict
import optax

from noisy_emcom.losses import listeners as listener_losses
from noisy_emcom.losses import speakers as speaker_losses
from noisy_emcom.networks import listeners, speakers
from noisy_emcom.utils import types
from noisy_emcom.utils import utils as emcom_utils
from noisy_emcom.utils.channels import channel_factory


class LewisGameAgent:
    """Plays a speaker/listener game with multi symbol."""

    def __init__(
        self,
        speaker: config_dict.ConfigDict,
        listener: config_dict.ConfigDict,
        channel: config_dict.ConfigDict,
        loss: config_dict.ConfigDict,
        opt: config_dict.ConfigDict,
        speaker_update_ema: float = 0.99,
    ) -> None:
        self._speaker_update_ema = speaker_update_ema

        # Prepares constructor.
        speaker = functools.partial(speakers.Speaker, **speaker)
        listener_cls = listeners.listener_factory(listener.listener_type)
        listener = functools.partial(listener_cls, **listener.kwargs)

        # hk.transform requires lambda to be built a posteriori in a pmap
        # pylint: disable=unnecessary-lambda
        self._speaker = hk.transform_with_state(
            lambda games, training_mode, actions_to_follow=None: speaker()(
                games=games,
                training_mode=training_mode,
                actions_to_follow=actions_to_follow,
            )
        )

        self._listener = hk.transform_with_state(
            lambda games, training_mode: listener()(
                games=games,
                training_mode=training_mode,
            )
        )
        # pylint: enable=unnecessary-lambda

        if loss.get("speaker", False):
            self._speaker_loss = speaker_losses.speaker_loss_factory(**loss.speaker)
        else:
            # We do not have speaker loss in EOL
            self._speaker_loss = None

        self._listener_loss = listener_losses.listener_loss_factory(**loss.listener)
        agents_opt = namedtuple("AgentsOpt", ["speaker", "listener"])
        self._agents_opt = agents_opt(
            speaker=emcom_utils.create_optimizer(opt.speaker),
            listener=emcom_utils.create_optimizer(opt.listener),
        )

        self._channel = channel_factory(**channel)

    @property
    def agents_opt(self):
        return self._agents_opt

    @property
    def agents_opt_init_fn(self):
        return {ag: getattr(self.agents_opt, ag).init for ag in self.agents_opt._fields}

    @property
    def params_type(self):
        return namedtuple(
            "Params",
            ["speaker", "target_speaker", "listener"],
            defaults=[None, None, None],
        )

    @property
    def states_type(self):
        return namedtuple(
            "States",
            ["speaker", "target_speaker", "listener"],
            defaults=[None, None, None],
        )

    @property
    def opt_states_type(self):
        return namedtuple("OptStates", ["speaker", "listener"], defaults=[None, None])

    @property
    def speaker(self):
        return self._speaker

    @property
    def listener_loss(self):
        return self._listener_loss

    def init(
        self,
        rng: chex.PRNGKey,
        init_games: types.DatasetInputs,
        training_mode: types.TrainingMode,
    ) -> tuple[types.Params, types.States]:
        """Returns speaker and listener params."""
        speaker_rng, listener_rng, channel_rng = jax.random.split(rng, num=3)
        games = self._init_games(init_games)
        params = self.params_type()
        states = self.states_type()

        games, params, states = self._init_speaker(
            speaker_rng, games, training_mode, params, states
        )

        games = games._replace(message=self._channel.add_noise(games, channel_rng))
        games, params, states = self._init_listener(
            listener_rng, games, training_mode, params, states
        )

        return params, states

    def unroll(
        self,
        params,
        states,
        global_step: int,
        rng: chex.PRNGKey,
        games: types.DatasetInputs,
        training_mode: types.TrainingMode,
    ) -> types.GamesData:
        """Unrolls the game for the forward pass."""
        # Prepares output.
        speaker_rng, listener_rng, channel_rng = jax.random.split(rng, num=3)
        games = types.GamesData(
            speaker_inp=games.speaker_inp,
            labels=games.labels,
            current_step=global_step,
        )

        # Step 1 : Speaker play.
        games = self._play_speaker(params, states, speaker_rng, games, training_mode)

        # Step 2 : Add noise to the message
        games = games._replace(message=self._channel.add_noise(games, channel_rng))

        # Step 3 : Listener play.
        games = self._play_listener(params, states, listener_rng, games, training_mode)

        return games

    def compute_loss(
        self, games: types.GamesData, rng: chex.PRNGKey
    ) -> types.AgentLossOutputs:
        """Computes Listener and Speaker losses."""

        # Computes listener loss and stats.
        listener_loss_outputs = self._listener_loss.compute_listener_loss(
            games=games, rng=rng
        )

        loss = listener_loss_outputs.loss
        stats = listener_loss_outputs.stats

        # Computes speaker loss and stats. (if necessary).
        if self._speaker_loss is not None:
            speaker_loss_outputs = self._speaker_loss.compute_speaker_loss(
                games=games,
                reward=listener_loss_outputs.reward,
            )

            loss += speaker_loss_outputs.loss
            stats.update(speaker_loss_outputs.stats)

        return types.AgentLossOutputs(
            loss=loss,
            reward=listener_loss_outputs.reward,
            probs=listener_loss_outputs.probs,
            stats=stats,
        )

    def optimize_agents(self, grads: chex.Array, params, opt_states):
        new_params = self.params_type()
        new_opt_states = self.opt_states_type()
        new_params, new_opt_states = self._optmize_speaker(
            grads, params, opt_states, new_params, new_opt_states
        )

        new_params, new_opt_states = self._optmizie_listener(
            grads, params, opt_states, new_params, new_opt_states
        )

        return new_params, new_opt_states

    def _init_games(self, init_games: types.DatasetInputs) -> types.GamesData:
        return types.GamesData(
            speaker_inp=init_games.speaker_inp, labels=init_games.labels, current_step=0
        )

    def _init_speaker(
        self,
        rng: chex.PRNGKey,
        games: types.GamesData,
        training_mode: types.TrainingMode,
        params,
        states,
    ) -> types.GamesData:
        speaker_rng, target_speaker_rng = jax.random.split(rng)
        params_speaker, states_speaker = self._speaker.init(
            speaker_rng,
            games=games,
            training_mode=training_mode,
        )

        speaker_outputs, _ = self._speaker.apply(
            params_speaker,
            states_speaker,
            speaker_rng,
            games=games,
            training_mode=training_mode,
        )

        params_target_speaker, states_target_speaker = self._speaker.init(
            target_speaker_rng,
            games=games,
            training_mode=types.TrainingMode.FORCING,
            actions_to_follow=speaker_outputs.action,
        )

        params = params._replace(
            speaker=params_speaker, target_speaker=params_target_speaker
        )

        states = states._replace(
            speaker=states_speaker, target_speaker=states_target_speaker
        )

        games = games._replace(
            speaker_outputs=speaker_outputs, message=speaker_outputs.action
        )

        return games, params, states

    def _init_listener(
        self,
        rng: chex.PRNGKey,
        games: types.GamesData,
        training_mode: types.TrainingMode,
        params: dict[str, hk.Params],
        states=dict[str, hk.State],
    ) -> tuple[types.GamesData, dict[str, hk.Params], dict[str, hk.State]]:
        params_listener, state_listener = self._listener.init(
            rng,
            games=games,
            training_mode=training_mode,
        )

        params = params._replace(listener=params_listener)
        states = states._replace(listener=state_listener)
        return games, params, states

    def _play_speaker(
        self,
        params,
        states,
        rng: chex.PRNGKey,
        games: types.GamesData,
        training_mode: types.TrainingMode,
    ) -> types.GamesData:
        speaker_outputs, _ = self._speaker.apply(
            params.speaker,
            states.speaker,
            rng,
            games=games,
            training_mode=training_mode,
        )

        target_speaker_outputs, _ = self._speaker.apply(
            params.target_speaker,
            states.target_speaker,
            rng,
            games=games,
            training_mode=types.TrainingMode.FORCING,
            actions_to_follow=speaker_outputs.action,
        )

        games = games._replace(
            speaker_outputs=speaker_outputs,
            target_speaker_outputs=target_speaker_outputs,
            message=speaker_outputs.action,
        )

        return games

    def _play_listener(
        self,
        params,
        states,
        rng: chex.PRNGKey,
        games: types.DatasetInputs,
        training_mode: types.TrainingMode,
    ) -> types.GamesData:
        listener_outputs, _ = self._listener.apply(
            params.listener,
            states.listener,
            rng,
            games=games,
            training_mode=training_mode,
        )

        games = games._replace(listener_outputs=listener_outputs)
        return games

    def _optmize_speaker(
        self,
        grads: chex.Array,
        params,
        opt_states,
        new_params,
        new_opt_states,
    ):
        speaker_opt_update = self.agents_opt.speaker.update

        speaker_updates, new_opt_state_speaker = speaker_opt_update(
            grads.speaker, opt_states.speaker, params.speaker
        )

        new_params_speaker = optax.apply_updates(params.speaker, speaker_updates)

        new_target_params = emcom_utils.update_target_params(
            rl_params=new_params_speaker,
            target_rl_params=params.target_speaker,
            target_network_update_ema=self._speaker_update_ema,
        )

        new_params = new_params._replace(
            speaker=new_params_speaker, target_speaker=new_target_params
        )

        new_opt_states = new_opt_states._replace(speaker=new_opt_state_speaker)
        return new_params, new_opt_states

    def _optmizie_listener(self, grads, params, opt_states, new_params, new_opt_states):
        listener_opt_update = self.agents_opt.listener.update
        listener_updates, new_opt_state_listener = listener_opt_update(
            grads.listener, opt_states.listener, params.listener
        )

        new_params_listener = optax.apply_updates(params.listener, listener_updates)
        new_params = new_params._replace(listener=new_params_listener)
        new_opt_states = new_opt_states._replace(listener=new_opt_state_listener)
        return new_params, new_opt_states
