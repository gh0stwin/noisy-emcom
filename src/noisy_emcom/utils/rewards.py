from typing import Any

import jax.numpy as jnp
from ml_collections import config_dict

from noisy_emcom.utils import types
from noisy_emcom.utils.scalers import scaler_factory
from noisy_emcom.utils.utils import create_scheduler


class Reward:
    def __init__(self, failure: float = 0.0, success: float = 1.0) -> None:
        assert success > failure
        self._failure = failure
        self._success = success

    @property
    def limits(self):
        return [self._failure, self._success]

    def failure(self, games: types.GamesData) -> float:
        return self._failure

    def success(self, games: types.GamesData) -> float:
        return self._success


def reward_factory(reward_type: types.GameRewardType, kwargs: types.Config):
    if reward_type == types.GameRewardType.DEFAULT:
        reward = Reward(**kwargs)
    else:
        raise ValueError(f"Incorrect reward type {reward_type}")

    return reward
