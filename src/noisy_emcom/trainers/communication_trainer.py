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

"""Helper to sample a population."""

import abc
import functools
from typing import Tuple

import chex
import jax
from jaxline import utils

from noisy_emcom.utils import population_storage as ps
from noisy_emcom.utils import types
from noisy_emcom.utils import utils as emcom_utils


@jax.pmap
def _split_keys_pmap(key):
    return tuple(jax.random.split(key))


class AbstractCommunicateTrainer(abc.ABC):
    """Abstract class implementation for training agents."""

    @abc.abstractmethod
    def communicate(
        self,
        global_step: int,
        rng: chex.PRNGKey,
        games: types.DatasetInputs,
        agent_storage: ps.PopulationStorage,
    ):
        pass


class BasicTrainer(AbstractCommunicateTrainer):
    """Sample trainer that simply loop over sampled agent pairs."""

    def __init__(
        self,
        update_fn,
        agents_count: dict[str, int],
        num_agents_per_step: int,
        training_mode: types.TrainingMode = types.TrainingMode.TRAINING,
    ) -> None:
        # Stores key values.
        self._agents_count = agents_count
        self._num_agents_per_step = num_agents_per_step

        # Prepares pmap functions.
        # Special pmap wrapper to correctly handle sampling across devices.
        self._pmap_sampling = emcom_utils.run_and_broadcast_to_all_devices(
            self._sample_fn
        )

        self._pmap_update_fn = jax.pmap(
            functools.partial(
                update_fn, training_mode=training_mode, is_sharded_update=True
            ),
            axis_name="i",
            donate_argnums=(0, 1, 2),
        )

    def _sample_fn(self, rng: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
        """Basic sampling function."""

        rngs = jax.random.split(rng, num=len(self._agents_count))
        agents_idx = {}

        for i, agent in enumerate(self._agents_count):
            agents_idx[agent] = jax.random.choice(
                key=rngs[i],
                a=self._agents_count[agent],
                replace=True,
                shape=[self._num_agents_per_step],
            )

        return agents_idx

    def communicate(
        self,
        global_step: chex.Array,
        rng: chex.PRNGKey,
        games: types.DatasetInputs,
        agent_storage: ps.PopulationStorage,
    ) -> Tuple[types.Config, ps.PopulationStorage]:
        """Performs one training step by looping over agent pairs."""

        # Step 1: samples the speaker/listener idx.
        sampling_rng, rng = _split_keys_pmap(rng)

        sampling_rng = utils.get_first(sampling_rng)

        agents_idx = self._pmap_sampling(sampling_rng)
        chex.assert_tree_shape_prefix(agents_idx, (self._num_agents_per_step,))

        # Step 2: executes a pmap update per speaker/listener pairs.
        scalars = None
        for i in agents_idx[list(agents_idx.keys())[0]]:
            # Next rng.
            update_rng, rng = _split_keys_pmap(rng)

            # Load agent params.
            current_idxs = {agent: idxs[i] for agent, idxs in agents_idx.items()}
            params, states, opt_states = agent_storage.load_society(
                agents_idx=current_idxs
            )

            # Performs update function (forward/backward pass).
            new_params, new_states, new_opt_states, scalars = self._pmap_update_fn(
                params,
                states,
                opt_states,
                global_step,
                games,
                update_rng,
            )

            # Updates params in storage.
            agent_storage.store_society(
                agents_idx=current_idxs,
                agents_params=new_params,
                agents_states=new_states,
                agents_opt_states=new_opt_states,
            )

        # Returns the scalar of the last random pair without the pmaped dimension.
        scalars = utils.get_first(scalars)

        return scalars, agent_storage
