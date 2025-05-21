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

"""Torso networks."""

from typing import Optional

import chex
import haiku as hk

from noisy_emcom.utils import types
from noisy_emcom.utils.utils import Identity


class DiscreteTorso(hk.Module):
    """Torso for discrete entries."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        mlp_kwargs: types.Config,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self._embedding = hk.Embed(vocab_size=vocab_size, embed_dim=embed_dim)
        self._mlp = hk.nets.MLP(**mlp_kwargs)

    def __call__(self, x: chex.Array, *args) -> chex.Array:
        h = self._embedding(x)
        return self._mlp(h)


class RecurrentTorso(hk.Module):
    def __init__(
        self, discrete_embed_kwargs: dict, rnn_kwargs: dict, name: Optional[str] = None
    ) -> None:
        super().__init__(name)
        self._discrete_embed = DiscreteTorso(**discrete_embed_kwargs)
        self._rnn = hk.LSTM(**rnn_kwargs)

    def __call__(self, x: chex.Array) -> chex.Array:
        embed_tokens = self._discrete_embed(x)
        initial_hidden = self._rnn.initial_state(embed_tokens.shape[0])
        hidden, _ = hk.static_unroll(
            self._rnn, embed_tokens, initial_hidden, time_major=False
        )

        return hidden[:, -1, :]


def torso_factory(
    torso_type: types.TorsoType,
    torso_kwargs: types.Config,
    name: str,
) -> hk.Module:
    """Builds torso from name and kwargs."""
    if torso_type == types.TorsoType.DISCRETE:
        torso = DiscreteTorso(name=name, **torso_kwargs)
    elif torso_type == types.TorsoType.MLP:
        torso = hk.nets.MLP(name=name, **torso_kwargs)
    elif torso_type == types.TorsoType.IDENTITY:
        torso = Identity(name=name)
    elif torso_type == types.TorsoType.RECURRENT:
        torso = RecurrentTorso(name=name, **torso_kwargs)
    else:
        raise ValueError(f"Incorrect torso type {torso_type}.")
    return torso
