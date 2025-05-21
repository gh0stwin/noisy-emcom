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

"""Defines different types of listeners."""

from typing import Union

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import optax

from noisy_emcom.networks import cores, heads, torsos
from noisy_emcom.utils import types
from noisy_emcom.utils.policy_actuator import policy_actuator_factory


class Listener(hk.Module):
    def __init__(
        self,
        torso_config: types.Config,
        core_config: types.Config,
        head_config: types.Config,
        task: types.Task,
        #discrimination_target: bool = False,
        name: str = "listener",
    ) -> None:
        super().__init__(name=name)
        self._torso = torsos.torso_factory(**torso_config, name="torso")
        self._core = cores.core_factory(**core_config, name="core")
        self._head = heads.head_factory(**head_config, name="head")
        self._task = task

        # Adding a dummy state to listeners to have symmetric speakers/listeners.
        hk.get_state("dummy_state", shape=(), init=hk.initializers.Constant(0.0))


class ListenerRl(Listener):
    """Reinforcement Learning (recurrent) Listener."""

    def __init__(
        self,
        torso_config: types.Config,
        core_config: types.Config,
        head_config: types.Config,
        policy_actuator_config: types.Config,
        temp_scheduler: str,
        temp_scheduler_kwargs: types.Config,
        task: types.Task,
        #discrimination_target: bool = False,
        name: str = "listener",
    ) -> None:
        super().__init__(
            torso_config, core_config, head_config, task, name
        )
        self._policy_actuator = policy_actuator_factory(**policy_actuator_config)
        self._temperature = getattr(optax, temp_scheduler)(**temp_scheduler_kwargs)

    def __call__(
        self, games: types.GamesData, training_mode: types.TrainingMode
    ) -> types.ListenerRlOutputs:
        """Unroll Listener over token of messages."""

        # Torso
        embedded_message = self._torso(games.message, games.round)

        # Core
        core_out = self._core(embedded_message)

        # Head
        if self._task == types.Task.DISCRIMINATION_DIST:
            target = games.labels[types.Task.DISCRIMINATION]["discr"]
        elif self._task == types.Task.CLASSIFICATION:
            target = games.labels[types.Task.CLASSIFICATION]
        elif self._task == types.Task.ATTRIBUTE:
            target = games.labels[types.Task.ATTRIBUTE]
        elif self._task == types.Task.RECONSTRUCTION:
            target = games.labels[types.Task.RECONSTRUCTION]
        else:
            target = games.speaker_inp

        head_outputs = self._head(core_out, target, training_mode)
        policy_logits = head_outputs.policy_logits
        temperatue = self._temperature(games.current_step)
        dist = distrax.Softmax(logits=policy_logits, temperature=temperatue)

        if training_mode == types.TrainingMode.TRAINING:
            rng = hk.next_rng_key()
            action = self._policy_actuator.act(dist, rng, games)
        elif training_mode in (
            types.TrainingMode.EVAL,
            types.TrainingMode.EVAL_LG,
            types.TrainingMode.EVAL_ILG,
        ):
            action = jnp.argmax(policy_logits, axis=-1)
        else:
            raise ValueError(f"Unknown training mode: {training_mode}.")

        return self._outputs(head_outputs, dist, action)

    def _outputs(
        self,
        head_outputs: types.RlHeadOutputs,
        action_dist: distrax.Distribution,
        sel_action: chex.Array,
    ) -> types.ListenerRlOutputs:
        policy_logits = head_outputs.policy_logits
        return types.ListenerRlOutputs(
            action=sel_action,
            action_log_prob=action_dist.log_prob(sel_action),
            entropy=action_dist.entropy(),
            policy_logits=policy_logits,
            probs=jax.tree_util.tree_map(
                jax.nn.softmax, jax.lax.stop_gradient(policy_logits)
            ),
            value=head_outputs.value,
            q_values=head_outputs.q_values,
        )


class ListenerSs(Listener):
    """Self-Supervised Listener."""

    def __call__(
        self, games: types.GamesData, training_mode: types.TrainingMode
    ) -> types.ListenerSsOutputs:
        """Unroll Listener over token of messages."""

        # Torso
        embedded_message = self._torso(games.message)

        # Core
        core_out = self._core(embedded_message)

        # Head
        if self._task == types.Task.DISCRIMINATION_DIST:
            target = games.labels[types.Task.DISCRIMINATION]["discr"]
        elif self._task == types.Task.CLASSIFICATION:
            target = games.labels[types.Task.CLASSIFICATION]
        elif self._task == types.Task.ATTRIBUTE:
            target = games.labels[types.Task.ATTRIBUTE]
        elif self._task == types.Task.RECONSTRUCTION:
            target = games.labels[types.Task.RECONSTRUCTION]
        else:
            target = games.speaker_inp

        listener_head_outputs = self._head(core_out, target, training_mode)
        return self._outputs(listener_head_outputs)

    def _outputs(
        self, head_outputs: types.ListenerSsHeadOutputs
    ) -> types.ListenerSsOutputs:
        return types.ListenerSsOutputs(
            predictions=head_outputs.predictions,
            targets=head_outputs.targets,
        )


def listener_factory(listener_type: str) -> Union[ListenerSs, ListenerRl]:
    """Retrieve listener class given a tag."""

    if listener_type == types.ListenerType.SS:
        listener = ListenerSs
    elif listener_type == types.ListenerType.RL:
        listener = ListenerRl
    else:
        raise ValueError(f"Incorrect listener type {listener_type}.")
    return listener
