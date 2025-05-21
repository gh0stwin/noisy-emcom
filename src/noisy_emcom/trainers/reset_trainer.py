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

"""Helper to sample listeners/speakers for resetting."""

from collections.abc import Callable, Iterable
import functools as fn
from typing import Union

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jaxline import utils

from noisy_emcom.utils import population_storage as ps
from noisy_emcom.utils import types


class ResetTrainer:
    """Class implementation for resetting speaker and listener agents."""

    def __init__(
        self,
        n_speakers: int,
        n_listeners: int,
        replace_n: int = 1,
        reset_opt: bool = True,
        components_to_replace: Union[Iterable[str], str] = "all",
    ):
        self._n_speakers = n_speakers
        self._n_listeners = n_listeners
        self._replace_n = replace_n
        self._change_props = ["params", "states"]
        self._reset_opt = reset_opt
        self._components_to_replace = components_to_replace

    def reset(
        self,
        rng: chex.PRNGKey,
        games: types.DatasetInputs,
        population_storage: ps.PopulationStorage,
        game_init_fn,
        opt_speaker_init_fn,
        opt_listener_init_fn,
        reset_type: types.ResetMode,
    ):
        """Implements random reset."""
        # Gets first then broadcasts to ensure same rng for all devices at init.
        rng = utils.get_first(rng)
        rng_speaker, rng_listener, rng = jax.random.split(rng, num=3)

        reset_listener_ids, reset_speaker_ids = [], []

        if reset_type == types.ResetMode.PAIR or reset_type == types.ResetMode.LISTENER:
            if self._replace_n >= self._n_listeners:
                reset_listener_ids = jnp.arange(self._n_listeners)
            else:
                reset_listener_ids = jax.random.randint(
                    key=rng_listener,
                    shape=(self._replace_n,),
                    minval=0,
                    maxval=self._n_listeners,
                )
        elif (
            reset_type == types.ResetMode.PAIR or reset_type == types.ResetMode.SPEAKER
        ):
            if self._replace_n >= self._n_speakers:
                reset_speaker_ids = jnp.arange(self._n_speakers)
            else:
                reset_speaker_ids = jax.random.randint(
                    key=rng_speaker,
                    shape=(self._replace_n,),
                    minval=0,
                    maxval=self._n_speakers,
                )

        rngs = list(
            jax.random.split(
                rng, num=max(len(reset_listener_ids), len(reset_speaker_ids))
            )
        )

        for i, split_rng in enumerate(rngs):
            rngs[i] = utils.bcast_local_devices(split_rng)

        population_storage = self._reset_agents(
            rngs,
            population_storage,
            reset_listener_ids,
            reset_speaker_ids,
            games,
            game_init_fn,
            opt_speaker_init_fn,
            opt_listener_init_fn,
        )

        return population_storage

    def _reset_agents(
        self,
        rngs: Iterable[chex.PRNGKey],
        population_storage: ps.PopulationStorage,
        listener_ids: Iterable[int],
        speaker_ids: Iterable[int],
        games: types.DatasetInputs,
        game_init_fn,
        opt_speaker_init_fn,
        opt_listener_init_fn,
    ) -> ps.PopulationStorage:
        reset_ids = {"listener": listener_ids, "speaker": speaker_ids}
        new_agents = []

        for i in range(max(len(listener_ids), len(speaker_ids))):
            new_agents.append(
                self._new_agent(
                    rngs[i],
                    games,
                    game_init_fn,
                    opt_speaker_init_fn,
                    opt_listener_init_fn,
                )
            )

        for agent_type, reset_agent_ids in reset_ids.items():
            for i, agent_id in enumerate(reset_agent_ids):
                population_storage = self._replace_agent(
                    population_storage,
                    agent_type,
                    agent_id,
                    new_agents[i][0],
                    new_agents[i][1],
                    new_agents[i][2],
                )

        return population_storage

    def _new_agent(
        self,
        rng_key: chex.PRNGKey,
        games: types.DatasetInputs,
        game_init_fn: Callable,
        opt_speaker_init_fn: Callable,
        opt_listener_init_fn: Callable,
    ) -> tuple:
        params_init_pmap = jax.pmap(
            fn.partial(
                game_init_fn,
                training_mode=types.TrainingMode.TRAINING,
            )
        )

        # Init Params/States.
        joint_params, joint_states = params_init_pmap(init_games=games, rng=rng_key)
        joint_opt_states = None

        if self._reset_opt:
            opt_speaker_init_pmap = jax.pmap(opt_speaker_init_fn)
            opt_listener_init_pmap = jax.pmap(opt_listener_init_fn)
            speaker_opt_states = opt_speaker_init_pmap(joint_params.speaker)
            listener_opt_states = opt_listener_init_pmap(joint_params.listener)

            joint_opt_states = types.OptStates(
                speaker=speaker_opt_states, listener=listener_opt_states
            )

        return joint_params, joint_states, joint_opt_states

    def _replace_agent(
        self,
        population_storage: ps.PopulationStorage,
        agent_type: str,
        agent_id: int,
        joint_params: types.Params,
        joint_states: types.States,
        joint_opt_states: types.OptStates,
    ) -> ps.PopulationStorage:
        if isinstance(self._components_to_replace, Iterable):
            joint_params, joint_states, joint_opt_states = self._replace_components(
                population_storage,
                agent_type,
                agent_id,
                joint_params,
                joint_states,
                joint_opt_states,
            )

        population_storage.store_agent(
            agent_type=agent_type,
            agent_idx=agent_id,
            params=getattr(joint_params, agent_type),
            states=getattr(joint_states, agent_type),
            opt_states=getattr(joint_opt_states, agent_type),
        )

        if agent_type == "speaker":
            population_storage.store_agent(
                agent_type="target_speaker",
                agent_idx=agent_id,
                params=joint_params.speaker,
                states=joint_states.speaker,
            )

        return population_storage

    def _replace_components(
        self,
        population_storage: ps.PopulationStorage,
        agent_type: str,
        agent_id: int,
        joint_params: types.Params,
        joint_states: types.States,
        joint_opt_states: types.OptStates,
    ):
        if agent_type == "listener":
            listener_props = population_storage.load_listener(agent_id)
            joint_params = self._reset_component_and_replace(
                listener_props, agent_type, "params", joint_params
            )

            joint_states = self._reset_component_and_replace(
                listener_props, agent_type, "states", joint_states
            )

            if not self._reset_opt:
                joint_opt_states = joint_opt_states._replace(
                    listener=listener_props.opt_states
                )
        else:
            speaker_props = population_storage.load_speaker(agent_id)
            joint_params = self._reset_component_and_replace(
                speaker_props, agent_type, "params", joint_params
            )

            joint_params = self._reset_component_and_replace(
                speaker_props, "target_speaker", "target_params", joint_params
            )

            joint_states = self._reset_component_and_replace(
                speaker_props, agent_type, "states", joint_states
            )

            joint_params = self._reset_component_and_replace(
                speaker_props, "target_speaker", "target_states", joint_params
            )

            if not self._reset_opt:
                joint_opt_states = joint_opt_states._replace(
                    speaker=speaker_props.opt_states
                )

        return joint_params, joint_states, joint_opt_states

    def _reset_component_and_replace(self, agent_props, agent_type, prop_name, module):
        partition = hk.data_structures.filter(
            lambda module_name, name, value: any(
                module in module_name for module in self._components_to_replace
            ),
            getattr(module, agent_type),
        )

        module = module._replace(
            **{
                agent_type: hk.data_structures.merge(
                    getattr(agent_props, prop_name), partition
                )
            }
        )

        return module
