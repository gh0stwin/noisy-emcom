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

"""Implement another experiment to compute ease of learning of agents."""

from typing import List, Optional

from absl import logging
import chex
import jax
import jax.numpy as jnp
from jaxline import utils, writers
from ml_collections import config_dict
import numpy as np

from noisy_emcom.data.dataset_consumer import DatasetConsumerVisitor
from noisy_emcom.experiments import lewis_experiment
from noisy_emcom.utils import checkpointer as ckpt_lib
from noisy_emcom.utils import types


class EaseOfLearningExperiment(lewis_experiment.LewisExperiment):
    """Ease of learning experiment.

    The ease of learning is defined as how fast a new listener acquires
    an emergent language (the speaker is fixed).
    """

    def __init__(
        self, mode: str, init_rng: chex.PRNGKey, config: config_dict.ConfigDict
    ) -> None:
        """Initializes experiment."""
        # Step 1: Loads ckpt and the related config.
        # Retrieve speaker params and config fo perform Ease of learning from a
        # given lewis configuration path
        ckpt = ckpt_lib.checkpointer_factory(
            config.checkpoint_experiment.type, config.checkpoint_experiment.kwargs
        )

        ckpt_data = ckpt.load_checkpoint(config.checkpointing.restore_path)
        ckpt.close()
        exp_state, lewis_cfg = ckpt_data.experiment_state, ckpt_data.config

        # Complete the eol configuration with lewis config option.
        config = config.unlock()
        config.agent.kwargs.speaker = lewis_cfg.agent.kwargs.speaker

        # Add dummy values that are required to start LewisExperiment
        config.agent.kwargs[
            "speaker_update_ema"
        ] = lewis_cfg.agent.kwargs.speaker_update_ema

        config = config.lock()

        # Step 2: Creates the lewis experiment to perform ease of learning.
        super().__init__(mode=mode, init_rng=init_rng, config=config)

        if mode == "train":
            # Overrides the speaker params with loaded ckpt.
            ckpt_params, ckpt_states = exp_state.params, exp_state.states
            speaker_params = ckpt_params["speaker"][config.speaker_index]
            speaker_states = ckpt_states["speaker"][config.speaker_index]
            self._population_storage.restore(
                params=dict(speaker=[speaker_params]),
                states=dict(speaker=[speaker_states]),
            )

    def train_loop(
        self,
        config: config_dict.ConfigDict,
        state,
        periodic_actions: List[utils.PeriodicAction],
        writer: Optional[writers.Writer] = None,
    ) -> None:
        """Overrides `train_loop` to collect the 'accuracy' output scalar values."""

        class CollectAccuracies:
            """A helper that collects 'accuracy' output scalar values."""

            def __init__(self) -> None:
                self.collector_accuracies = []

            def update_time(self, t: float, step: int) -> None:
                del t, step  # Unused.

            def __call__(
                self,
                t: float,
                step: int,
                scalar_outputs: types.Config,
            ) -> None:
                del t, step  # Unused.
                self.collector_accuracies.append(scalar_outputs["accuracy"])

        collector = CollectAccuracies()
        # Weirdly type(periodic_actions) is tuple and not list!
        super().train_loop(
            config=config,
            state=state,
            periodic_actions=list(periodic_actions) + [collector],
            writer=writer,
        )

        # Fetches from device and stack the accuracy numbers.
        accuracies = np.array(jax.device_get(collector.collector_accuracies))
        logging.info("Ease of learning accuracies per listener %s", accuracies)

    def evaluate(
        self, global_step: chex.ArrayNumpy, rng: chex.PRNGKey, **unused_kwargs
    ) -> types.Config:
        info = {}
        original, reconstruct = [], []
        data = self._data_builder.get_evaluation_data(
            DatasetConsumerVisitor(), self._mode.split("_")[1]
        )

        params, states, _ = self._population_storage.load_society(
            agents_idx=dict(speaker=0, listener=0)
        )

        params = utils.get_first(params)
        states = utils.get_first(states)
        num_games, sum_scalars = 0, None

        for samples in data:
            agent_outputs, games = self._eval_batch(
                params=params,
                states=states,
                games=samples,
                rng=rng,
                current_step=global_step,
                agent=self._agent,
                training_mode=self._config.training_mode,
            )

            num_games += games.speaker_inp.shape[0]

            if sum_scalars is None:
                sum_scalars = agent_outputs.stats
            else:
                sum_scalars = jax.tree_util.tree_map(
                    jnp.add, agent_outputs.stats, sum_scalars
                )

            if self._config.task == types.Task.RECONSTRUCTION and num_games < 1024:
                original += [np.array(games.listener_outputs.targets)]
                reconstruct += [np.array(games.listener_outputs.predictions)]

        if self._config.task == types.Task.RECONSTRUCTION:
            info["images"] = dict(
                original=np.concatenate(original, axis=0),
                reconstruct=np.concatenate(reconstruct, axis=0),
            )

        avg_scalars = jax.tree_util.tree_map(lambda x: x / num_games, sum_scalars)
        avg_scalars = jax.device_get(avg_scalars)
        logging.info("Eval [Step %d] %s", global_step, avg_scalars)
        return avg_scalars, info
