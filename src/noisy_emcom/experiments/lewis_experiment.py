# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Updated by www.github.com/gh0stwin

"""Emergent Communication jaxline experiment."""

from typing import List, NamedTuple, Optional, Tuple

from absl import flags, logging
import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jaxline import utils, writers
from ml_collections import config_dict
import numpy as np
import optax

from noisy_emcom.agents.agent_factory import agent_factory
from noisy_emcom.agents.lewis_game_agent import LewisGameAgent
from noisy_emcom.data.dataset_consumer import DatasetConsumerVisitor
from noisy_emcom.data.dataset_loaders import dataset_loader_factory
from noisy_emcom.trainers import (
    communication_trainer,
    imitation_trainer,
    reset_opt_trainer,
    reset_trainer,
)
from noisy_emcom.utils import experiment_with_checkpointing as jaxline_ckpt
from noisy_emcom.utils import language_measures
from noisy_emcom.utils import population_storage as ps
from noisy_emcom.utils import types
from noisy_emcom.utils import utils as emcom_utils

# This file should only include langame and jaxline dependencies!
FLAGS = flags.FLAGS


# A split helper that operates on a pmap-ed rng_key.
@jax.pmap
def _split_three_keys_pmap(key):
    return tuple(jax.random.split(key, num=3))


class LewisExperiment(jaxline_ckpt.ExperimentWithCheckpointing):
    """Cidre experiment.

    Note: we here inherit from ExperimentWithCheckpointing to abstract the ckpt
    mechanism that is entangled in jaxline.
    Beware that  ExperimentWithCheckpointing inherits from
    experiment.AbstractExperiment in jaxline.
    """

    def __init__(
        self, mode: str, init_rng: chex.PRNGKey, config: config_dict.ConfigDict
    ) -> None:
        """Initializes experiment."""
        super().__init__(mode=mode, init_rng=init_rng, config=config)

        self._mode = mode
        data_rng, pop_rng, self._init_rng = jax.random.split(init_rng, num=3)

        # By default, we do not use a population
        pop_config = self._config.population
        self._agents_count = pop_config.get("agents_count")
        self._num_agents_per_step = pop_config.get("num_agents_per_step", 1)

        # Prepares games.
        self._agent = agent_factory(
            self._config.agent.agent_type, self._config.agent.kwargs
        )

        self._data_builder = dataset_loader_factory.get(
            config=self._config.game,
            train_batch_size=self._config.training.batch_size,
            eval_batch_size=self._config.evaluation.batch_size,
        )

        # Prepares parameters.
        self._population_storage = ps.PopulationStorage(
            params_type=self._agent.params_type,
            states_type=self._agent.states_type,
            opt_states_type=self._agent.opt_states_type,
            agents_count=self._agents_count,
        )

        # Train vs. Eval.
        if self._mode == "train":
            self._train_data_consumer = self._data_builder.get_training_data(
                DatasetConsumerVisitor(), data_rng
            )

            # Lewis trainer
            # Constructs the trainer that sample and update agents pairs.
            self._communication_trainer = communication_trainer.BasicTrainer(
                update_fn=self._update_fn,
                agents_count=self._agents_count,
                num_agents_per_step=self._num_agents_per_step,
                training_mode=types.TrainingMode(self._config.training_mode),
            )

            # Imitation trainer.
            if self._config.imitation and self._config.imitation.imitation_step:
                # Checks config values.
                if (
                    not self._config.imitation.self_imitation
                    and self._agents_count["speaker"] < 2
                ):
                    raise ValueError(
                        "Invalid imitation config: n_speaker must be larger."
                        " than one."
                    )
                if (
                    self._config.imitation.self_imitation
                    and self._agents_count["speaker"] != 1
                ):
                    raise ValueError(
                        "Invalid imitation config: n_speaker must be equal"
                        " to one for self-imitation."
                    )

                # Cases where we perform imitation training.
                logging.info("Training option: apply imitation.")
                self._imitation_trainer = imitation_trainer.ImitateTrainer(
                    n_speakers=self._agents_count["spaker"],
                    imitation_update_fn=self._imitation_update_fn,
                )
            else:
                # Cases where we do not perform imitation training.
                logging.info("Training option: Do not apply imitation.")
                self._imitation_trainer = None

            # Resets trainer.
            if config.reset and self._config.reset.reset_step:
                logging.info("Training option: apply resetting.")
                self._reset_trainer = reset_trainer.ResetTrainer(
                    n_speakers=self._agents_count["speaker"],
                    n_listeners=self._agents_count["listener"],
                )
            else:
                # Cases where we do not perform resetting.
                logging.info("Training option: Do not apply resetting.")
                self._reset_trainer = None

            if self._config.reset_opt and self._config.reset_opt.reset_step > 0:
                self._reset_opt_trainer = reset_opt_trainer.ResetOptTrainer()
            else:
                self._reset_opt_trainer = None

            # Initializes network/optim param/states.
            self._population_storage.initialize(
                rng=pop_rng,
                games=next(self._train_data_consumer),
                game_init_fn=self._agent.init,
                agents_opt_init_fn=self._agent.agents_opt_init_fn,
                opt_states_cls=self._agent.opt_states_type,
            )

            self._train_data_consumer.reset()
        else:
            self._eval_batch = jax.jit(
                self.eval_batch, static_argnames=("agent", "training_mode")
            )

            self._communication_trainer = None
            self._imitation_trainer = None
            self._reset_trainer = None
            self._reset_opt_trainer = None
            self._eval_topsim_mode = self._config.evaluation.get(
                "eval_topsim_mode", False
            )

    #  _             _
    # | |_ _ __ __ _(_)_ __
    # | __| '__/ _` | | '_ \
    # | |_| | | (_| | | | | |
    #  \__|_|  \__,_|_|_| |_|
    #

    def step(
        self,
        *,
        global_step: chex.ArrayNumpy,
        rng: chex.PRNGKey,
        writer: Optional[writers.Writer],
    ) -> types.Config:
        """A single training step."""
        games = next(self._train_data_consumer)

        rng_communicate, rng_imitate, rng_reset = _split_three_keys_pmap(rng)

        # Performs one step of population training.
        # Population trainer sample agents pair before `_update_func` per pair.
        scalars, self._population_storage = self._communication_trainer.communicate(
            global_step=global_step,
            rng=rng_communicate,
            games=games,
            agent_storage=self._population_storage,
        )

        global_step = utils.get_first(global_step)

        # Imitation learning every imitation_step steps.
        if (
            self._imitation_trainer
            and global_step > 0
            and global_step % self._config.imitation.imitation_step == 0
        ):
            imit_scalar, self._population_storage = self._imitation_trainer.imitate(
                rng=rng_imitate,
                games=games,
                agent_storage=self._population_storage,
                **self._config.imitation,
            )
            scalars.update(imit_scalar)

        # Reset step.
        if (
            self._reset_trainer
            and global_step > 0
            and global_step % self._config.reset.reset_step == 0
        ):
            self._population_storage = self._reset_trainer.reset(
                rng=rng_reset,
                games=games,
                population_storage=self._population_storage,
                game_init_fn=self._agent.init,
                opt_speaker_init_fn=self._agent.agents_opt.speaker.init,
                opt_listener_init_fn=self._agent.agents_opt.listener.init,
                reset_type=self._config.reset.reset_type,
            )

        if (
            self._reset_opt_trainer
            and global_step > 0
            and global_step == self._config.reset_opt.reset_step
        ):
            self._population_storage = self._reset_opt_trainer.reset(
                self._population_storage,
                opt_speaker_init_fn=self._agent.agents_opt.speaker.init,
                opt_listener_init_fn=self._agent.agents_opt.listener.init,
                reset_mode=self._config.reset_opt.reset_mode,
            )

        if global_step == self._config.training.steps - 1:
            logging.info("Training [Final Step %d] %s", global_step, scalars)

        # Returns the scalar of the last random pair.
        return scalars

    def _update_fn(
        self,
        params: types.Params,
        states: types.States,
        opt_states: types.OptStates,
        global_step: int,
        games: types.DatasetInputs,
        rng: chex.PRNGKey,
        training_mode: types.TrainingMode,
        is_sharded_update: bool = True,
    ) -> Tuple[types.Params, types.States, types.OptStates, types.Config]:
        """Applies an update to parameters and returns new state.

        Args:
          params: The current (speaker, listener) params to update.
          states: The current (speaker, listener) states.
          opt_states: The current optimizer state for speaker and listener.
          games: The input batch of games to learn on.
          rng: The random key.
          training_mode: defines the training_mode (TRAIN=sampling, EVAL=greedy).
          is_sharded_update: If set, the code assumes it's running within the
            context of a pmap, and thus would use jax.lax.pxxx functions to average
            gradients or measurementes across chips/shards.

        Returns:
          new_params: The updated params.
          new_states: The updated state.
          new_opt_states: The updated optimizer state.
          scalars: A dict of scalar measurements to log.
        """
        grad_loss_fn = jax.grad(self._loss_fn, has_aux=True)
        grads, agent_loss_stats = grad_loss_fn(
            params,
            states=states,
            games=games,
            global_step=global_step,
            rng=rng,
            training_mode=training_mode,
        )

        if is_sharded_update:
            # grad_loss_fn outputs the grads divided by the number of devices
            # (jax.device_count()). We apply the psum to get the mean across devices.
            grads = jax.lax.psum(grads, axis_name="i")

        new_params, new_opt_states = self._agent.optimize_agents(
            grads, params, opt_states
        )

        # Scalars to log (note: we log the mean across all hosts/devices).
        scalars = jax.tree_util.tree_map(
            lambda x: x / games.speaker_inp.shape[0], agent_loss_stats
        )

        if is_sharded_update:
            scalars = jax.lax.pmean(scalars, axis_name="i")

        # Stores the score of the individual speakers inside the state
        # Retrieves speaker states.
        speaker_state = states.speaker
        counter = speaker_state["speaker"]["counter"]
        avg_score = speaker_state["speaker"]["avg_score"]

        # Updates speaker by computing the average score.
        mutable_state = hk.data_structures.to_mutable_dict(speaker_state)
        mutable_state["speaker"]["avg_score"] = (counter * avg_score) / (
            counter + 1
        )  # + scalars["global_accuracy"] / (counter + 1)
        mutable_state["speaker"]["counter"] += 1
        speaker_state = hk.data_structures.to_haiku_dict(mutable_state)

        # Updates states across devices.
        speaker_state = jax.lax.pmean(speaker_state, axis_name="i")
        new_states = states._replace(speaker=speaker_state)

        return new_params, new_states, new_opt_states, scalars

    def _loss_fn(
        self,
        params: types.Params,
        states: types.States,
        games: types.DatasetInputs,
        global_step: int,
        rng: chex.PRNGKey,
        training_mode: types.TrainingMode,
    ):
        rng_unroll, rng_loss = jax.random.split(rng)
        games_data = self._agent.unroll(
            params,
            states,
            global_step=global_step,
            rng=rng_unroll,
            games=games,
            training_mode=training_mode,
        )
        agent_loss_outputs = self._agent.compute_loss(games=games_data, rng=rng_loss)

        def avg_fn(value, games):
            return value / games.speaker_inp.shape[0]

        scaled_loss = avg_fn(agent_loss_outputs.loss, games) / jax.device_count()
        return scaled_loss, agent_loss_outputs.stats

    def _imitation_update_fn(
        self,
        games: types.DatasetInputs,
        params_student: hk.Params,
        params_oracle: hk.Params,
        state_student: hk.State,
        state_oracle: hk.State,
        opt_state: optax.OptState,
        rng: chex.PRNGKey,
    ):
        # Gets labels (output of the oracle).
        games = types.GamesData(speaker_inp=games.speaker_inp, labels=games.labels)
        # rng not used as training_mode=EVAL.
        oracle_outputs, _ = self._agent.speaker.apply(
            params_oracle,
            state_oracle,
            rng,
            games=games,
            training_mode=types.TrainingMode.EVAL,
        )
        # Computes gradient.
        grad_supervised_loss_fn = jax.grad(self._supervised_loss_fn, has_aux=True)
        scaled_grads, loss = grad_supervised_loss_fn(
            params_student,
            state_student,
            labels=jax.lax.stop_gradient(oracle_outputs.action),
            games=games,
            rng=rng,
        )
        grads = jax.lax.psum(scaled_grads, axis_name="i")

        # Computes and applies updates via our optimizer.
        speaker_opt_update = self._agent.agents_opt.speaker.update
        speaker_updates, new_opt_state_speaker = speaker_opt_update(grads, opt_state)
        new_params_speaker = optax.apply_updates(params_student, speaker_updates)

        # Scalars to log (note: we log the mean across all hosts/devices).
        scalars = loss / games.speaker_inp.shape[0]
        scalars = jax.lax.pmean(scalars, axis_name="i")

        return new_params_speaker, state_student, new_opt_state_speaker, scalars

    def _supervised_loss_fn(
        self,
        params_student: hk.Params,
        state_student: hk.Params,
        labels: chex.Array,
        games: types.DatasetInputs,
        rng: chex.PRNGKey,
    ):
        prediction_outputs, _ = self._agent.speaker.apply(
            params_student,
            state_student,
            rng,
            games=games,
            training_mode=types.TrainingMode.TRAINING,
        )
        logits = jnp.transpose(prediction_outputs.policy_logits, [0, 2, 1])
        labels = jax.nn.one_hot(
            labels, self._config.speaker.vocab_size, dtype=logits.dtype
        )
        # [B, T]
        loss = emcom_utils.softmax_cross_entropy(logits, labels)
        # Average on T and sum on B
        loss = jnp.sum(jnp.mean(loss, axis=-1), axis=0)

        def avg_fn(x):
            return x / logits.shape[0]

        scaled_loss = avg_fn(loss) / jax.device_count()

        return scaled_loss, loss

    def _next_batch(self, global_step: int):
        return self._train_data_consumer.next_batch(global_step)

    #                  _
    #   _____   ____ _| |
    #  / _ \ \ / / _` | |
    # |  __/\ V / (_| | |
    #  \___| \_/ \__,_|_|
    #

    def evaluate(
        self,
        *,
        global_step: chex.ArrayNumpy,
        rng: chex.PRNGKey,
        writer: Optional[writers.Writer] = None,
    ) -> types.Config:
        """See base class."""

        # Gives a mode equal to either test or valid.
        # Gives a ensemble_type in [vote, average].
        _, mode, ensemble_type = self._mode.split("_")

        # Computes metrics over the evaluation games.
        if self._eval_topsim_mode:
            game_scalars, messages = self._eval_topsim(
                global_step, mode, ensemble_type, rng
            )
        else:
            game_scalars, messages = self._eval_over_games(
                global_step, mode, ensemble_type, rng
            )

        # Computes metrics by iterating over concepts.
        # It is only computed over speaker message, independently of listener.
        message_scalars = {}  # self._eval_over_messages(messages)

        # Fuses and formats scalars.
        scalars = {**game_scalars, **message_scalars}

        scalars = jax.device_get(scalars)

        logging.info("Eval [Step %d] %s", global_step, scalars)

        return scalars

    def _eval_over_games(
        self, global_step: int, mode: str, ensemble_type: str, rng: chex.PRNGKey
    ) -> Tuple[types.Config, List[chex.Array]]:
        # Eval at most the self._config.evaluation.max_n_agents first agents.
        n_speakers = np.min(
            [self._agents_count["speaker"], self._config.evaluation.max_n_agents]
        )

        n_listeners = np.min(
            [self._agents_count["listener"], self._config.evaluation.max_n_agents]
        )

        # Initializes values.
        num_games, sum_scalars = 0, None
        topographic_similarity = {task: [] for task in self._config.evaluation.topsim}
        messages = [[] for _ in range(n_speakers)]

        # Prepares subsampling.
        subsampling_ratio = self._config.evaluation.subsampling_ratio
        assert 0.01 <= subsampling_ratio <= 1
        data = self._data_builder.get_evaluation_data(DatasetConsumerVisitor(), mode)

        for samples in data:
            for speaker_id in range(n_speakers):
                all_agents_outputs = []
                for listener_id in range(n_listeners):
                    ensemble_scalars = {}

                    # Retrieves params.
                    params, states, _ = self._population_storage.load_society(
                        agents_idx=dict(speaker=speaker_id, listener=listener_id)
                    )

                    params = utils.get_first(params)  # eval is on single device only
                    states = utils.get_first(states)  # eval is on single device only

                    # Play game.
                    # rng is not used at eval time.
                    agent_outputs, games = self._eval_batch(
                        params=params,
                        states=states,
                        games=samples,
                        rng=rng,
                        current_step=global_step,
                        agent=self._agent,
                        training_mode=self._config.training_mode,
                    )

                    all_agents_outputs.append(agent_outputs)

                # Computes scalar by averaging all listeners.
                if n_listeners > 1:
                    ensemble_scalars = self._eval_all_listeners(
                        ensemble_type=ensemble_type,
                        predictions=all_agents_outputs,
                        games=games,
                    )

                # Saves ensemble stats and stats for the last listener (one pair).
                scalars = {**ensemble_scalars, **agent_outputs.stats}

                # Updates counters.
                num_games += games.speaker_inp.shape[0]

                # Accumulates the sum of scalars for each step.
                if sum_scalars is None:
                    sum_scalars = scalars
                else:
                    sum_scalars = jax.tree_util.tree_map(jnp.add, scalars, sum_scalars)

                # Computes message statistics. As it is independent of the listener,
                # we here arbitrary take the last listener.
                # slices = max(10, int(games.speaker_inp.shape[0] * subsampling_ratio))
                # top_sim = types.TopSimData(
                #     message=games.message,
                #     input_meaning=games.speaker_inp,
                #     label_meaning=games.labels,
                # )

                # # Takes only the first slices examples.
                # slice_top_sim = jax.tree_util.tree_map(
                #     lambda x, y=slices: x[:y], top_sim
                # )

                # for sim, sim_values in self._config.evaluation.topsim.items():
                #     meaning = sim_values["meaning"]
                #     task = sim_values["task"]
                #     topographic_similarity[sim] += [
                #         language_measures.games_topographic_similarity(
                #             top_sim_data=slice_top_sim, meaning_sim=meaning, task=task
                #         )
                #     ]

                # Stores message for end-game analysis.
                messages[speaker_id].append(games.speaker_outputs.action)

        # Averages per number of total games (both wrt batches and populations).
        avg_scalars = jax.tree_util.tree_map(lambda x: x / num_games, sum_scalars)

        # for sim in self._config.evaluation.topsim:
        #     avg_scalars[f"topsim_{sim}"] = np.mean(topographic_similarity[sim])

        # stacks messages into a single batch.
        messages = [np.concatenate(m, axis=0) for m in messages]
        return avg_scalars, messages

    def _eval_topsim(
        self, global_step: int, mode: str, ensemble_type: str, rng: chex.PRNGKey
    ) -> Tuple[types.Config, List[chex.Array]]:
        # Eval at most the self._config.evaluation.max_n_agents first agents.
        n_speakers = np.min(
            [self._agents_count["speaker"], self._config.evaluation.max_n_agents]
        )

        n_listeners = np.min(
            [self._agents_count["listener"], self._config.evaluation.max_n_agents]
        )

        # Initializes values.
        num_games, sum_scalars = 0, None
        topographic_similarity = {task: [] for task in self._config.evaluation.topsim}
        messages = [[] for _ in range(n_speakers)]

        # Prepares subsampling.
        subsampling_ratio = self._config.evaluation.subsampling_ratio
        assert 0.01 <= subsampling_ratio <= 1
        data = self._data_builder.get_evaluation_data(DatasetConsumerVisitor(), mode)

        for i, samples in enumerate(data):
            for speaker_id in range(n_speakers):
                all_agents_outputs = []

                for listener_id in range(n_listeners):
                    ensemble_scalars = {}

                    # Retrieves params.
                    params, states, _ = self._population_storage.load_society(
                        agents_idx=dict(speaker=speaker_id, listener=listener_id)
                    )

                    params = utils.get_first(params)  # eval is on single device only
                    states = utils.get_first(states)  # eval is on single device only

                    # Play game.
                    # rng is not used at eval time.
                    agent_outputs, games = self._eval_batch(
                        params=params,
                        states=states,
                        games=samples,
                        rng=rng,
                        current_step=global_step,
                        agent=self._agent,
                        training_mode=self._config.training_mode,
                    )

                    all_agents_outputs.append(agent_outputs)

                # Computes scalar by averaging all listeners.
                if n_listeners > 1:
                    ensemble_scalars = self._eval_all_listeners(
                        ensemble_type=ensemble_type,
                        predictions=all_agents_outputs,
                        games=games,
                    )

                # Saves ensemble stats and stats for the last listener (one pair).
                scalars = {**ensemble_scalars, **agent_outputs.stats}

                # Updates counters.
                num_games += games.speaker_inp.shape[0]

                # Accumulates the sum of scalars for each step.
                if sum_scalars is None:
                    sum_scalars = scalars
                else:
                    sum_scalars = jax.tree_util.tree_map(jnp.add, scalars, sum_scalars)

                # Computes message statistics. As it is independent of the listener,
                # we here arbitrary take the last listener.
                slices = max(10, int(games.speaker_inp.shape[0] * subsampling_ratio))
                top_sim = types.TopSimData(
                    message=games.message,
                    input_meaning=games.speaker_inp,
                    label_meaning=games.labels,
                )

                # # Takes only the first slices examples.
                slice_top_sim = jax.tree_util.tree_map(
                    lambda x, y=slices: x[:y], top_sim
                )

                for sim, sim_values in self._config.evaluation.topsim.items():
                    meaning = sim_values["meaning"]
                    task = sim_values["task"]
                    topographic_similarity[sim] += [
                        language_measures.games_topographic_similarity(
                            top_sim_data=slice_top_sim, meaning_sim=meaning, task=task
                        )
                    ]

                # Stores message for end-game analysis.
                messages[speaker_id].append(games.speaker_outputs.action)

        # Averages per number of total games (both wrt batches and populations).
        # avg_scalars = jax.tree_util.tree_map(lambda x: x / num_games, sum_scalars)
        avg_scalars = {}

        for sim in self._config.evaluation.topsim:
            avg_scalars[f"topsim_{sim}"] = np.mean(topographic_similarity[sim])

        # stacks messages into a single batch.
        messages = [np.concatenate(m, axis=0) for m in messages]
        return avg_scalars, messages

    def _eval_all_listeners(
        self,
        ensemble_type: str,
        predictions: List[types.AgentLossOutputs],
        games: types.GamesData,
    ):
        if ensemble_type == "vote":
            probs = [x.probs for x in predictions]
            # Stacks leaves of probs, which can be a list of dictionaries for classif.
            stacked_pred = jax.tree_util.tree_map(
                lambda *vals: np.stack(vals, axis=-1), *probs
            )  # [B, F, listeners]

            avg_prediction = jax.tree_util.tree_map(
                lambda x: jnp.mean(x, axis=-1), stacked_pred
            )  # [B, F]

            ensemble_pred = jax.tree_util.tree_map(
                lambda x: jnp.argmax(x, axis=-1), avg_prediction
            )  # [B]

            scalars = self._agent.listener_loss.compute_ensemble_accuracy(
                prediction=ensemble_pred, games=games
            )
        elif ensemble_type == "average":
            accuracies = jnp.array([x.stats["global_accuracy"] for x in predictions])
            scalars = dict(ensemble_acc=jnp.mean(accuracies))
        else:
            raise ValueError(f"Wrong ensemble type: {ensemble_type}.")

        return scalars

    def _eval_over_messages(self, messages: List[chex.Array]) -> types.Config:
        # Computes edit distance between messages from different speakers.
        edit_distance = []
        message_per_games = np.stack(messages, axis=1)  # [n_games, n_speaker, T]
        for message in message_per_games:
            # These messages are from the same game, and thus encode the same concept.
            edit_dist = language_measures.edit_dist(message)
            edit_dist = np.mean(edit_dist)
            edit_distance.append(edit_dist)

        return dict(edit_distance=np.mean(edit_distance))

    def eval_batch(
        self,
        params: NamedTuple,
        states: NamedTuple,
        games: types.DatasetInputs,
        rng: chex.PRNGKey,
        current_step: int,
        agent: LewisGameAgent,
        training_mode: types.TrainingMode,
    ) -> Tuple[types.AgentLossOutputs, types.GamesData]:
        game_rng, loss_rng = jax.random.split(rng)
        finished_game = agent.unroll(
            params,
            states,
            current_step,
            rng=game_rng,
            games=games,
            training_mode=types.TrainingMode(training_mode),
        )

        agent_loss_outputs = agent.compute_loss(games=finished_game, rng=loss_rng)
        return agent_loss_outputs, finished_game
