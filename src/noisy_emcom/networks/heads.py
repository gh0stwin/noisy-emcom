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

"""Head networks."""
import math
from typing import Iterable, Optional

import chex
import haiku as hk
import jax
import jax.numpy as jnp

from noisy_emcom.utils import types
from noisy_emcom.utils.utils import (
    activation_function_factory,
    cosine_loss,
    is_test_mode,
)


class MultiMlpHead(hk.Module):
    """MultiMLP head."""

    def __init__(
        self,
        hidden_sizes: Iterable[int],
        task: types.Task,
        name: Optional[str] = "multi_mlp_head",
    ):
        super().__init__(name)
        self._hidden_sizes = tuple(hidden_sizes)
        self._task = task

    def __call__(
        self,
        message_repr: chex.Array,
        target: chex.Array,
        training_mode: types.TrainingMode,
    ) -> types.ListenerSsHeadOutputs:
        mlps = jax.tree_util.tree_map(
            lambda x: hk.nets.MLP(output_sizes=self._hidden_sizes + (x.shape[-1],)),
            target,
        )

        predictions = jax.tree_util.tree_map(lambda x, m=message_repr: x(m), mlps)
        return types.ListenerSsHeadOutputs(predictions=predictions, targets=target)


class PolicyHead(hk.Module):
    """Policy head."""

    def __init__(
        self, num_actions: int, hidden_sizes: Iterable[int], name: Optional[str] = None
    ) -> None:
        super().__init__(name)
        self._policy_head = hk.nets.MLP(
            output_sizes=tuple(hidden_sizes) + (num_actions,)
        )

    def __call__(self, inputs) -> types.SpeakerHeadOutputs:
        return types.SpeakerHeadOutputs(policy_logits=self._policy_head(inputs))


class DuelingHead(hk.Module):
    """Dueling value head."""

    def __init__(
        self, num_actions: int, hidden_sizes: Iterable[int], name: Optional[str] = None
    ) -> None:
        super().__init__(name)
        self._value_net = hk.nets.MLP(tuple(hidden_sizes) + (1,))
        self._advantage_net = hk.nets.MLP(tuple(hidden_sizes) + (num_actions,))

    def __call__(self, inputs) -> types.DuelingHeadOutputs:
        state_value = self._value_net(inputs)
        advantage = self._advantage_net(inputs)
        mean_advantage = jnp.mean(advantage, axis=-1, keepdims=True)
        q_values = state_value + advantage - mean_advantage
        return types.DuelingHeadOutputs(q_values=q_values, value=state_value)


class SigmoidDuelingHead(hk.Module):
    """Dueling value head."""

    def __init__(
        self, num_actions: int, hidden_sizes: Iterable[int], name: Optional[str] = None
    ) -> None:
        super().__init__(name)
        self._value_net = hk.nets.MLP(tuple(hidden_sizes) + (1,))
        self._advantage_net = hk.nets.MLP(tuple(hidden_sizes) + (num_actions,))

    def __call__(self, inputs) -> types.DuelingHeadOutputs:
        state_value = jax.nn.sigmoid(self._value_net(inputs))
        advantage = jax.nn.sigmoid(self._advantage_net(inputs))
        mean_advantage = jnp.mean(advantage, axis=-1, keepdims=True)
        q_values = state_value + advantage - mean_advantage
        return types.DuelingHeadOutputs(q_values=q_values, value=state_value)


class PolicyValueHead(hk.Module):
    """Policy and Value-function head."""

    def __init__(
        self, num_actions: int, hidden_sizes: Iterable[int], name: Optional[str] = None
    ) -> None:
        super().__init__(name)
        self._policy_head = hk.nets.MLP(tuple(hidden_sizes) + (num_actions,))
        self._value_head = hk.nets.MLP(tuple(hidden_sizes) + (1,))

    def __call__(self, inputs) -> types.RlHeadOutputs:
        return types.SpeakerHeadOutputs(
            policy_logits=self._policy_head(inputs), value=self._value_head(inputs)
        )



class DistancePolicyValueHead(hk.Module):
    def __init__(
        self,
        message_hidden_sizes: Iterable[int],
        candidates_hidden_sizes: Iterable[int],
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self._message_encoder = hk.nets.MLP(tuple(message_hidden_sizes))
        self._candidates_encoder = hk.nets.MLP(tuple(candidates_hidden_sizes))
        self._value_head = hk.nets.MLP((1,))

    def __call__(
        self, message_repr: chex.Array, games: types.GamesData
    ) -> types.RlHeadOutputs:
        message_repr = self._message_encoder(message_repr)
        candidates = self._candidates_encoder(games.speaker_inp)
        logits = 1 - cosine_loss(message_repr[:, None, :], candidates[None, :, :]) / 2
        state_value = jax.nn.sigmoid(self._value_head(logits))
        return types.RlHeadOutputs(policy_logits=logits, value=state_value)



class MergePolicyValueHead(hk.Module):
    """Policy and Value-function head with two inputs (merged before passing input to
    heads).

    """

    def __init__(
        self,
        message_hidden_sizes: Iterable[int],
        games_hidden_sizes: Iterable[int],
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self._message_encoder = hk.nets.MLP(tuple(message_hidden_sizes))
        self._games_encoder = hk.nets.MLP(tuple(games_hidden_sizes))
        self._value_head = hk.nets.MLP((1,))

    def __call__(
        self, message_repr: chex.Array, games: types.GamesData
    ) -> types.RlHeadOutputs:
        message_repr = self._message_encoder(message_repr)
        candidates = self._games_encoder(games.speaker_inp)
        features = jnp.matmul(message_repr, candidates.transpose([1, 0]))
        value_features = self._value_head(features)
        return types.RlHeadOutputs(policy_logits=features, value=value_features)


class NormalizeMergePolicyValueHead(hk.Module):
    """Policy and Value-function head with two inputs (merged and normalize before
    passing input to heads).

    """

    def __init__(
        self,
        message_hidden_sizes: Iterable[int],
        games_hidden_sizes: Iterable[int],
        value_net_size: Iterable[int] = (1,),
        hidden_act_func: str = "tanh",
        value_act_func: str = "tanh",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self._message_encoder = hk.nets.MLP(tuple(message_hidden_sizes))
        self._games_encoder = hk.nets.MLP(tuple(games_hidden_sizes))
        self._hidden_act = activation_function_factory(hidden_act_func)
        self._value_head = hk.nets.MLP(value_net_size)
        self._value_act = activation_function_factory(value_act_func)

    def __call__(
        self,
        message_repr: chex.Array,
        target: chex.Array,
        training_mode: types.TrainingMode,
    ) -> types.RlHeadOutputs:
        message_repr = self._hidden_act(self._message_encoder(message_repr))
        candidates = self._hidden_act(self._games_encoder(target))
        features = jnp.matmul(message_repr, candidates.transpose([1, 0]))

        if is_test_mode(training_mode):
            shape = features.shape[:-1] + (1,)
            value_features = jnp.zeros(shape, dtype=features.dtype)
        else:
            value_features = self._value_act(self._value_head(features))

        return types.RlHeadOutputs(policy_logits=features, value=value_features)


class PolicyQValueHead(hk.Module):
    """Policy and Qvalue head."""

    def __init__(
        self, num_actions: int, hidden_sizes: Iterable[int], name: Optional[str] = None
    ) -> None:
        super().__init__(name)
        self._policy_head = hk.nets.MLP(
            output_sizes=tuple(hidden_sizes) + (num_actions,)
        )
        self._value_head = hk.nets.MLP(
            output_sizes=tuple(hidden_sizes) + (num_actions,)
        )
        self._q_value_head = DuelingHead(
            num_actions=num_actions, hidden_sizes=hidden_sizes
        )

    def __call__(self, inputs) -> types.SpeakerHeadOutputs:
        dueling_head_outputs = self._q_value_head(inputs)
        return types.SpeakerHeadOutputs(
            policy_logits=self._policy_head(inputs),
            q_values=dueling_head_outputs.q_values,
            value=dueling_head_outputs.value,
        )


class PolicyQValueDuelingHead(hk.Module):
    """Policy and Qvalue head."""

    def __init__(
        self, num_actions: int, hidden_sizes: Iterable[int], name: Optional[str] = None
    ) -> None:
        super().__init__(name)
        self._policy_head = hk.nets.MLP(
            output_sizes=tuple(hidden_sizes) + (num_actions,)
        )
        self._value_head = hk.nets.MLP(output_sizes=tuple(hidden_sizes) + (1,))
        self._q_value_head = DuelingHead(
            num_actions=num_actions, hidden_sizes=hidden_sizes
        )

    def __call__(self, inputs) -> types.SpeakerHeadOutputs:
        return types.SpeakerHeadOutputs(
            policy_logits=self._policy_head(inputs),
            q_values=self._q_value_head(inputs).q_values,
            value=self._value_head(inputs),
        )


class CpcHead(hk.Module):
    """CPC head."""

    def __init__(
        self, hidden_sizes: Iterable[int], name: Optional[str] = "cpc_head"
    ) -> None:
        super().__init__(name)
        self.proj_pred = hk.nets.MLP(output_sizes=hidden_sizes)
        self.proj_target = hk.nets.MLP(output_sizes=hidden_sizes)

    def __call__(
        self, message_repr: chex.Array, target: chex.Array
    ) -> types.ListenerSsHeadOutputs:
        # Takes the second view if it exist, otherwise, takes same input view.
        # if types.Task.DISCRIMINATION in games.labels:
        #     target_inputs = games.labels[types.Task.DISCRIMINATION]
        # else:
        #     target_inputs = games.speaker_inp

        return types.ListenerSsHeadOutputs(
            predictions=jax.nn.tanh(self.proj_pred(message_repr)),
            targets=jax.nn.tanh(self.proj_target(target)),
        )


class ReconstructionHead(hk.Module):
    def __init__(
        self,
        latent_size: Iterable = (4, 4, 128),
        upsample_factor: int = 2,
        name: Optional[str] = "reconstruction_head",
    ):
        super().__init__(name)
        self._latent_size = latent_size
        self._upsample_factor = upsample_factor
        self._embedding = hk.Linear(math.prod(self._latent_size))
        self._convs = [
            hk.Conv2D(64, 3, padding="SAME", with_bias=False),
            hk.Conv2D(32, 3, padding="SAME", with_bias=False),
            hk.Conv2D(16, 3, padding="SAME", with_bias=False),
            hk.Conv2D(16, 3, padding="SAME", with_bias=False),
        ]

        self._final_conv = hk.Conv2D(3, 3, padding="SAME", with_bias=False)

    def __call__(
        self,
        message_repr: chex.Array,
        target: chex.Array,
        training_mode: types.TrainingMode,
    ) -> types.EaseOfLearningOutputs:
        # assert types.Task.RECONSTRUCTION in games.labels

        out = message_repr
        out = self._embedding(out)
        out = out.reshape((-1,) + self._latent_size)
        out = jax.nn.relu(out)

        for conv in self._convs:
            new_shape = list(out.shape)
            new_shape[-2] *= self._upsample_factor
            new_shape[-3] *= self._upsample_factor
            out = jax.image.resize(out, new_shape, jax.image.ResizeMethod.NEAREST)
            out = conv(out)
            out = jax.nn.relu(out)

        out = self._final_conv(out)
        out = jax.nn.tanh(out)
        return types.EaseOfLearningOutputs(
            predictions=out, targets=target
        )


class DiscriminationHead(hk.Module):
    def __init__(
        self,
        hidden_sizes: Iterable[int],
        activation_func: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self._task = types.Task.DISCRIMINATION
        self._hidden_sizes = hidden_sizes
        self._embed_mess = hk.nets.MLP(output_sizes=hidden_sizes)
        self._activation = activation_function_factory(activation_func)

    def __call__(
        self,
        message_repr: chex.Array,
        target: chex.Array,
        training_mode: types.TrainingMode,
    ):
        hidden_mess = self._embed_mess(message_repr)
        hidden_mess = self._activation(hidden_mess)
        cand_mlps = jax.tree_util.tree_map(
            lambda x: hk.nets.MLP(output_sizes=self._hidden_sizes), target
        )

        hidden_cand = jax.tree_util.tree_map(
            lambda mlp, x: self._activation(mlp(x)), cand_mlps, target
        )

        targets = jax.tree_util.tree_map(
            lambda x: jax.nn.one_hot(jnp.arange(x.shape[0]), x.shape[0]), target
        )

        return types.EaseOfLearningOutputs(
            predictions=jax.tree_util.tree_map(
                lambda x: jnp.matmul(hidden_mess, x.transpose([1, 0])), hidden_cand
            ),
            targets=targets,
        )


class DiscriminationDistanceHead(hk.Module):
    def __init__(
        self,
        hidden_sizes: Iterable[int],
        activation_func: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self._task = types.Task.DISCRIMINATION
        self._embed_mess = hk.nets.MLP(output_sizes=hidden_sizes)
        self._embed_targets = hk.nets.MLP(output_sizes=hidden_sizes)
        self._activation_func = activation_function_factory(activation_func)

    def __call__(
        self,
        message_repr: chex.Array,
        target: chex.Array,
        training_mode: types.TrainingMode,
    ):
        hidden_mess = self._embed_mess(message_repr)
        hidden_mess = self._activation_func(hidden_mess)
        hidden_targets = self._embed_targets(target)
        hidden_targets = self._activation_func(hidden_targets)
        return types.EaseOfLearningOutputs(
            predictions=hidden_mess, targets=hidden_targets
        )


def head_factory(
    head_type: str,
    head_kwargs: types.Config,
    kwargs: types.Config,
    name: str,
) -> hk.Module:
    """Builds head from name and kwargs."""

    loss_specific_kwargs = kwargs.get(head_type, dict())
    all_kwargs = {**head_kwargs, **loss_specific_kwargs}

    if head_type == types.SpeakerHeadType.POLICY:
        head = PolicyHead(name=name, **all_kwargs)
    elif head_type == types.SpeakerHeadType.POLICY_VALUE:
        head = PolicyValueHead(name=name, **all_kwargs)
    elif head_type == types.ListenerHeadType.DIST_POLICY_VALUE:
        head = DistancePolicyValueHead(name=name, **all_kwargs)
    elif head_type == types.SpeakerHeadType.POLICY_QVALUE:
        head = PolicyQValueHead(name=name, **all_kwargs)
    elif head_type == types.SpeakerHeadType.POLICY_QVALUE_DUELING:
        head = PolicyQValueDuelingHead(name=name, **all_kwargs)
    elif head_type == types.ListenerHeadType.MULTIMLP:
        head = MultiMlpHead(name=name, **all_kwargs)
    elif head_type == types.ListenerHeadType.CPC:
        head = CpcHead(name=name, **all_kwargs)
    elif head_type == types.ListenerHeadType.MERGE_POLICY_VALUE:
        head = MergePolicyValueHead(name=name, **all_kwargs)
    elif head_type == types.ListenerHeadType.NORM_MERGE_POL_VAL:
        head = NormalizeMergePolicyValueHead(name=name, **all_kwargs)
    elif head_type == types.ListenerHeadType.DISCRIMINATION:
        head = DiscriminationHead(name=name, **all_kwargs)
    elif head_type == types.ListenerHeadType.DISCRIMINATION_DIST:
        head = DiscriminationDistanceHead(name=name, **all_kwargs)
    elif head_type == types.ListenerHeadType.RECONSTRUCTION:
        head = ReconstructionHead(name=name, **all_kwargs)
    else:
        raise ValueError(f"Incorrect head type {head_type}.")
    return head
