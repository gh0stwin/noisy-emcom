import chex
import distrax
import jax
import jax.numpy as jnp

from noisy_emcom.utils import types
from noisy_emcom.utils.utils import create_scheduler


class PolicyActuator:
    def act(
        self, dist: distrax.Distribution, rng: chex.PRNGKey, games: types.GamesData
    ) -> chex.Array:
        return dist.sample(seed=rng)


class EpsilonGreedyPolicyActuator(PolicyActuator):
    def __init__(self, scheduler: str, scheduler_kwargs: types.Config) -> None:
        super().__init__()
        self._epsilon_schedule = create_scheduler(scheduler, scheduler_kwargs)

    def act(
        self, dist: distrax.Categorical, rng: chex.PRNGKey, games: types.GamesData
    ) -> chex.Array:
        rng_greedy, rng_exploit, rng_dist = jax.random.split(rng, num=3)
        batch_shape = dist.batch_shape
        epsilon = self._epsilon_schedule(games.current_step)
        return jnp.where(
            jax.random.uniform(rng_greedy, batch_shape) < epsilon,
            jax.random.randint(rng_exploit, batch_shape, 0, dist.num_categories),
            dist.sample(seed=rng_dist),
        )


def policy_actuator_factory(
    policy_actuator_type: types.PolicyActuatorType, kwargs: types.Config
):
    if policy_actuator_type == types.PolicyActuatorType.DEFAULT:
        pol_act = PolicyActuator(**kwargs)
    elif policy_actuator_type == types.PolicyActuatorType.EPSILON_GREEDY:
        pol_act = EpsilonGreedyPolicyActuator(**kwargs)
    else:
        raise ValueError(f"Incorrect policy actuator type {policy_actuator_type}")

    return pol_act
