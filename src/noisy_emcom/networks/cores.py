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

"""Reccurent networks."""

from typing import Optional, Sequence, Tuple

import chex
import haiku as hk
import jax.numpy as jnp

from noisy_emcom.utils import types


class CustomedIdentityCore(hk.RNNCore):
    """A recurrent core that forwards the inputs and a mock state.

    This is commonly used when switching between recurrent and feedforward
    versions of a model while preserving the same interface.
    """

    def __init__(
        self,
        hidden_size: int,
        name: Optional[str] = None,
    ) -> None:
        """Constructs an CustomedIdentityCore.

        Args:
          hidden_size: Hidden layer size.
          name: Name of the module.
        """
        super().__init__(name=name)
        self.hidden_size = hidden_size

    def __call__(
        self,
        inputs: Sequence[chex.Array],
        state: hk.LSTMState,
    ) -> Tuple[Sequence[chex.Array], hk.LSTMState]:
        return inputs, state

    def initial_state(self, batch_size: Optional[int]) -> hk.LSTMState:
        return hk.LSTMState(
            hidden=jnp.zeros([batch_size, self.hidden_size]),
            cell=jnp.zeros([batch_size, self.hidden_size]),
        )


class SingleInputAttention(hk.MultiHeadAttention):
    def __init__(
        self,
        num_heads: int,
        key_size: int,
        w_init: Optional[hk.initializers.Initializer] = None,
        value_size: Optional[int] = None,
        model_size: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(num_heads, key_size, 0, value_size, model_size, name)
        self.w_init = w_init

        if self.w_init is None:
            self.w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")

    def __call__(
        self,
        query: jnp.ndarray,
        key: Optional[jnp.ndarray] = None,
        value: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        key = key if key is not None else query
        value = value if value is not None else key
        out = super().__call__(query, key, value, mask)
        return out.sum(axis=-2)


class StaticLstm(hk.Module):
    def __init__(self, hidden_size: int, name: Optional[str] = None):
        super().__init__(name)
        self._lstm = hk.LSTM(hidden_size)

    def __call__(self, x: chex.Array) -> chex.Array:
        initial_state = self._lstm.initial_state(x.shape[0])
        core_out, _ = hk.static_unroll(self._lstm, x, initial_state, time_major=False)
        return core_out[:, -1, :]  # Only consider the last repr. of core


class RnnWithAttention(hk.Module):
    def __init__(self, hidden_size: int, name: Optional[str] = None):
        super().__init__(name)
        self._lstm = hk.LSTM(hidden_size)
        self._attn = LocalAttention()

    def __call__(self, x: chex.Array) -> chex.Array:
        initial_state = self._lstm.initial_state(x.shape[0])
        out, _ = hk.static_unroll(self._lstm, x, initial_state, time_major=False)
        out = self._attn(out)
        return out


class ToCoreState(hk.Module):
    """Module to get a core state from an embedding."""

    def __init__(
        self,
        prototype: types.RNNState,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self._prototype = prototype

    def __call__(self, embedding: chex.Array) -> types.RNNState:
        if isinstance(self._prototype, hk.LSTMState):
            return _ToLSTMState(self._prototype.cell.shape[-1])(embedding)
        elif isinstance(self._prototype, chex.Array):
            return hk.Linear(output_size=self._prototype.shape[-1])(embedding)
        elif not self._prototype:
            return ()
        else:
            raise ValueError(
                f"Invalid prototype type for core state " f"{type(self._prototype)}."
            )


class _ToLSTMState(hk.Module):
    """Module linearly mapping a tensor to an hk.LSTMState."""

    def __init__(self, output_size: int) -> None:
        super().__init__(name="to_lstm_state")
        self._linear = hk.Linear(output_size=2 * output_size)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        h, c = jnp.split(self._linear(inputs), indices_or_sections=2, axis=-1)
        return hk.LSTMState(h, c)


def core_factory(
    core_type: types.CoreType,
    core_kwargs: types.Config,
    name: str,
) -> hk.RNNCore:
    """Builds core from name and kwargs."""
    if core_type == types.CoreType.LSTM:
        core = hk.LSTM(name=name, **core_kwargs)
    elif core_type == types.CoreType.GRU:
        core = hk.GRU(name=name, **core_kwargs)
    elif core_type == types.CoreType.STATIC_RNN:
        core = StaticLstm(name=name, **core_kwargs)
    elif core_type == types.CoreType.IDENTITY:
        core = CustomedIdentityCore(name=name, **core_kwargs)
    elif core_type == types.CoreType.ATTENTION:
        core = SingleInputAttention(name=name, **core_kwargs)
    elif core_type == types.CoreType.RNN_ATTENTION:
        core = RnnWithAttention(name=name, **core_kwargs)
    else:
        raise ValueError(f"Incorrect core type {core_type}.")
    return core
