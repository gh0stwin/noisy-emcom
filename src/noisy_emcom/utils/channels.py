from typing import Any, Callable, Union

import chex
import jax
import jax.numpy as jnp

from noisy_emcom.utils import types
from noisy_emcom.utils.utils import create_scheduler


class Channel:
    def add_noise(self, games: types.GamesData, rng: chex.PRNGKey) -> chex.Array:
        return games.message


class NoisyChannel(Channel):
    def __init__(self, scheduler: str, scheduler_kwargs: dict[str, Any]) -> None:
        super().__init__()
        self._scheduler = create_scheduler(scheduler, scheduler_kwargs)

    def add_noise(self, games: types.GamesData, rng: chex.PRNGKey) -> chex.Array:
        template_rng, noise_rng, rng = jax.random.split(rng, num=3)
        message = super().add_noise(games, rng)
        noise_template = self._noise_message_template(message.shape, template_rng)
        noise_likelihood = self._scheduler(games.current_step)
        return jnp.where(
            jax.random.uniform(noise_rng, message.shape) < noise_likelihood,
            noise_template,
            message,
        )

    def _noise_message_template(
        self, message_shape: tuple, rng: chex.PRNGKey
    ) -> Union[int, chex.Array]:
        raise NotImplementedError()


class DetectableNoisyChannel(NoisyChannel):
    def __init__(
        self, unknown_token: int, scheduler: Callable, scheduler_kwargs: dict[str, Any]
    ) -> None:
        super().__init__(scheduler, scheduler_kwargs)
        self._unknown_token = unknown_token

    def _noise_message_template(
        self, message_shape: tuple, rng: chex.PRNGKey
    ) -> Union[int, chex.Array]:
        return self._unknown_token


class DetectableFixedTokensNoisyChannel(DetectableNoisyChannel):
    def __init__(
        self, unknown_token: int, message_length: int, mask_tokens: tuple[int]
    ):
        self._unknown_token = unknown_token
        self._mask = jnp.zeros((message_length,), dtype=bool)

        if len(mask_tokens) > 0:
            self._mask = self._mask.at[jnp.array(mask_tokens)].set(True)

    def add_noise(self, games: types.GamesData, rng: chex.PRNGKey) -> chex.Array:
        mess = games.message
        mess = jnp.where(
            self._mask, self._noise_message_template(mess.shape, rng), mess
        )

        return mess


class IndistinctNoisyChannel(NoisyChannel):
    def __init__(
        self, vocab_size: int, scheduler: Callable, scheduler_kwargs: dict[str:Any]
    ) -> None:
        super().__init__(scheduler, scheduler_kwargs)
        self._vocab_size = vocab_size

    def _noise_message_template(
        self, message_shape: tuple, rng: chex.PRNGKey
    ) -> Union[int, chex.Array]:
        return jax.random.randint(rng, message_shape, 0, self._vocab_size)


def channel_factory(channel_type: str, channel_kwargs: types.Config) -> Channel:
    if channel_type == types.ChannelType.DEFAULT:
        channel = Channel(**channel_kwargs)
    elif channel_type == types.ChannelType.DETECTABLE:
        channel = DetectableNoisyChannel(**channel_kwargs)
    elif channel_type == types.ChannelType.DETECTABLE_FIXED_TOKENS:
        channel = DetectableFixedTokensNoisyChannel(**channel_kwargs)
    elif channel_type == types.ChannelType.INDISTINCT:
        channel = IndistinctNoisyChannel(**channel_kwargs)
    else:
        raise ValueError(f"Incorrect channel type: {channel_type}")

    return channel
