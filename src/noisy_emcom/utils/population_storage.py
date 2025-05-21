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

"""Creates a class to store and load the param of a population of agents."""

import functools as fn
from typing import NamedTuple, Optional

import chex
import haiku as hk
import jax
from jaxline import utils
import optax

from noisy_emcom.utils import types


class PopulationStorage:
    """Stores the population params and states."""

    def __init__(
        self, params_type, states_type, opt_states_type, agents_count: dict[str, int]
    ) -> None:
        self._params_type = params_type
        self._states_type = states_type
        self._opt_states_type = opt_states_type
        self._agents_count = agents_count
        self._init_storage()

    @property
    def params(self):
        return self._params

    @property
    def states(self):
        return self._states

    @property
    def opt_states(self):
        return self._opt_states

    def load_agent(
        self, agent_type: str, agent_idx: int
    ) -> tuple[hk.Params, hk.State, optax.OptState]:
        """Load single agent (normal or target agent)."""
        params = self._params[agent_type][agent_idx]
        states = self._states[agent_type][agent_idx]
        target_type = self._target_type(agent_type)
        target_params = None
        target_states = None
        opt_states = None

        if target_type in self._params:
            target_params = self._params[target_type][agent_idx]

        if target_type in self._states:
            target_states = self._states[target_type][agent_idx]

        if agent_type in self._opt_states:
            opt_states = self._opt_states[agent_type][agent_idx]

        return types.AgentProperties(
            params=params,
            states=states,
            target_params=target_params,
            target_states=target_states,
            opt_states=opt_states,
        )

    def store_agent(
        self,
        agent_type: str,
        agent_idx: int,
        params: Optional[hk.Params] = None,
        states: Optional[hk.State] = None,
        opt_states: Optional[optax.OptState] = None,
    ) -> None:
        """Once data for an agent is updated, store it back."""
        if params:
            self._params[agent_type][agent_idx] = params

        if states:
            self._states[agent_type][agent_idx] = states

        if opt_states:
            self._opt_states[agent_type][agent_idx] = opt_states

    # Checkpointing utilities.
    def snapshot(self) -> types.AllProperties:
        return types.AllProperties(
            params=self._params,
            states=self._states,
            opt_states=self._opt_states,
        )

    def restore(
        self,
        params: Optional[types.AllParams] = None,
        states: Optional[types.AllStates] = None,
        opt_states: Optional[types.AllOptStates] = None,
    ) -> None:
        """Restores all params/states of the agent/optimizer."""
        if params:
            self._params.update(params)

        if states:
            self._states.update(states)

        if opt_states:
            self._opt_states.update(opt_states)

    def initialize(
        self,
        rng: chex.PRNGKey,
        games: types.DatasetInputs,
        game_init_fn,
        agents_opt_init_fn,
        opt_states_cls,
    ) -> None:
        """Initializes all the params/states of the agent/optimizer."""
        # Initializes params, states & opt states.
        self._init_storage()

        # Prepares per agents pmap function.
        params_init_pmap = jax.pmap(
            fn.partial(game_init_fn, training_mode=types.TrainingMode.TRAINING)
        )

        # Iterates over speaker/listener options.
        for agent_type, count in self._agents_count.items():
            opt_init_pmap = jax.pmap(agents_opt_init_fn[agent_type])

            for i in range(count):
                # Prepares rng.
                agent_rng, rng = jax.random.split(rng)
                agent_rng = utils.bcast_local_devices(agent_rng)

                # Init Params/States.
                joint_params, joint_states = params_init_pmap(
                    init_games=games, rng=agent_rng
                )

                joint_opt_states = opt_init_pmap(getattr(joint_params, agent_type))
                self.store_society(
                    agents_idx={agent_type: i},
                    agents_params=joint_params,
                    agents_states=joint_states,
                    agents_opt_states=opt_states_cls(**{agent_type: joint_opt_states}),
                )

    def load_society(
        self, agents_idx: dict[str, int]
    ) -> tuple[NamedTuple, NamedTuple, NamedTuple]:
        """Prepares params and opt_states for a given pair of speaker/listener."""
        society = {}
        params, states, opt_states = {}, {}, {}

        for agent_type, idx in agents_idx.items():
            society[agent_type] = self.load_agent(agent_type, idx)

        for agent_type, props in society.items():
            params[agent_type] = props.params
            states[agent_type] = props.states

            if props.opt_states is not None:
                opt_states[agent_type] = props.opt_states

            if props.target_params is not None:
                params[self._target_type(agent_type)] = props.target_params

            if props.target_states is not None:
                states[self._target_type(agent_type)] = props.target_states

        return (
            self._params_type(**params),
            self._states_type(**states),
            self._opt_states_type(**opt_states),
        )

    def store_society(
        self,
        agents_idx: dict[str, int],
        agents_params,
        agents_states,
        agents_opt_states,
    ) -> None:
        """Once data for a pair speaker/listener is updated, store it back."""
        for agent_type, idx in agents_idx.items():
            self.store_agent(
                agent_type,
                idx,
                getattr(agents_params, agent_type),
                getattr(agents_states, agent_type),
                getattr(agents_opt_states, agent_type),
            )

            target_type = self._target_type(agent_type)
            self.store_agent(
                target_type,
                idx,
                getattr(agents_params, target_type, None),
                getattr(agents_states, target_type, None),
                getattr(agents_opt_states, target_type, None),
            )

    def _init_storage(self):
        params = {agent: [None] * count for agent, count in self._agents_count.items()}
        params |= {
            self._target_type(agent): [None] * count
            for agent, count in self._agents_count.items()
            if self._target_type(agent) in self._params_type._fields
        }

        states = {agent: [None] * count for agent, count in self._agents_count.items()}
        states |= {
            self._target_type(agent): [None] * count
            for agent, count in self._agents_count.items()
            if self._target_type(agent) in self._states_type._fields
        }

        opt_states = {
            agent: [None] * count for agent, count in self._agents_count.items()
        }

        opt_states |= {
            self._target_type(agent): [None] * count
            for agent, count in self._agents_count.items()
            if self._target_type(agent) in self._opt_states_type._fields
        }

        self._params: types.AllParams = params
        self._states: types.AllStates = states
        self._opt_states: types.AllOptStates = opt_states

    def _target_type(self, agent_type: str) -> str:
        return "target_" + agent_type
