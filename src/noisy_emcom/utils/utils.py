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

"""Utility functions for CIDRE experiments."""

from collections.abc import Iterable
import functools
import operator
from typing import Callable, Optional, Tuple, Union

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jaxline import utils
from ml_collections import config_dict
import numpy as np
import optax

from noisy_emcom.utils import types
from noisy_emcom.utils.scalers import Constant


def activation_function_factory(function: Optional[str] = None):
    if function == "relu":
        return jax.nn.relu
    elif function == "tanh":
        return jax.nn.tanh
    elif function == "sigmoid":
        return jax.nn.sigmoid

    return Constant()


def default_features(default_type="norm"):
    data = np.load("data/processed/default-features.npy")
    if default_type == "zero":
        return data[0]
    elif default_type == "one":
        return data[1]
    elif default_type == "half":
        return data[2]
    elif default_type == "norm":
        return data[3]

    raise ValueError(f"Incorrect default logit type {default_type}.")


def create_scheduler(
    scheduler: Union[str, Iterable[str]],
    scheduler_kwargs: Union[types.Config, Iterable[types.Config]],
):
    """Create single scheduler or a sequence of schedules."""
    if isinstance(scheduler, (list, tuple)):
        assert isinstance(scheduler_kwargs, (list, tuple))
        assert len(scheduler) == len(scheduler_kwargs)
        schedulers = []
        boundaries = []

        for i, kwargs in enumerate(scheduler_kwargs):
            kwargs = kwargs.copy()
            if i > 0:
                boundaries.append(int(kwargs["transition_begin"]))
                kwargs["transition_begin"] = 0

            schedulers.append(create_scheduler(scheduler[i], kwargs))

        return optax.join_schedules(schedulers, boundaries)

    _create_scheduler = getattr(optax, scheduler)
    return _create_scheduler(**scheduler_kwargs)


def softmax_cross_entropy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
) -> jnp.ndarray:
    """Computes softmax cross entropy given logits and one-hot class labels.

    Args:
      logits: Logit output values.
      labels: Ground truth one-hot-encoded labels.

    Returns:
      Loss value that has the same shape as `labels`.
    """
    loss = -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
    return loss


def l2_normalize(
    x: chex.Array,
    axis: Optional[int] = None,
    epsilon: float = 1e-12,
) -> chex.Array:
    """l2 normalize a tensor on an axis with numerical stability."""
    square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
    x_inv_norm = jax.lax.rsqrt(jnp.maximum(square_sum, epsilon))
    return x * x_inv_norm


def cosine_loss(x: chex.Array, y: chex.Array) -> chex.Array:
    """CPC's regression loss. This is a simple cosine distance."""
    normed_x, normed_y = l2_normalize(x, axis=-1), l2_normalize(y, axis=-1)
    return jnp.sum((normed_x - normed_y) ** 2, axis=-1)


def update_target_params(
    rl_params: hk.Params,
    target_rl_params: hk.Params,
    target_network_update_ema: float,
) -> hk.Params:
    """Maybe update target params."""

    new_target_rl_params = optax.incremental_update(
        new_tensors=rl_params,
        old_tensors=target_rl_params,
        step_size=1 - target_network_update_ema,
    )

    return new_target_rl_params


def is_test_mode(mode: types.TrainingMode) -> bool:
    return mode in (types.TrainingMode.EVAL_LG, types.TrainingMode.EVAL_ILG)


def global_to_local_pmap(data: chex.ArrayTree) -> chex.ArrayTree:
    """From a tensor for all/global devices get the slice for local devices.

    Args:
      data: arbitrarily nested structure of tensors, all having a leading
        dimension of the size of all/global devices.

    Returns:
      The slice of each tensor that correspond to local devices.
    """
    chex.assert_tree_shape_prefix(data, (jax.device_count(),))
    start, end = _global_to_local_pmap_indexes()
    return jax.tree_util.tree_map(lambda t: t[start:end], data)


def _global_to_local_pmap_indexes() -> Tuple[int, int]:
    """An internal helper for global_to_local_pmap."""
    start, end = 0, 0
    for process_index in range(jax.process_count()):
        process_devices = jax.local_devices(process_index=process_index)
        end = start + len(process_devices)
        if process_index == jax.process_index():
            break
        start = end
    assert end > start
    return start, end


def run_and_broadcast_to_all_devices(fn: Callable[[chex.PRNGKey], chex.ArrayTree]):
    """Given a one-process sampling fn, wrap it for multi-process setup.

    Sampling in a multi-process and multi TPU-chips context is not easy.
    This helper takes a vanilla sampler that assumes exactly one process
    is sampling for all other processes, and wraps it to run in a multi-process,
    multi-TPU context, to get benefit of TPU communication to pass the sampling
    results to all processes.

    Args:
      fn: The callable that does sampling in a non-parallel context (no pmap). It
        expects a single (not batched, not pmaped) `rng_key`.

    Returns:
      A callable that correctly samples using pmap and distributes the results
      across all TPU chips/shards/processes.
    """
    # The trick is to only use one process if we are in a multi-process setup.

    @functools.partial(jax.pmap, axis_name="s")
    @jax.util.wraps(fn, namestr="sample({fun})")
    def _pmap_run_fn(rng_key, is_current_chip_the_one: bool):
        """Runs `fn` on all chips, but uses only results where `onehot_mask`."""
        samples = fn(rng_key)
        # Apply the given mask, which should be a boolean scalar.
        samples = jax.tree_util.tree_map(lambda t: t * is_current_chip_the_one, samples)
        # Transfer the samples from the main process to all other ones.
        samples = jax.lax.psum(samples, axis_name="s")
        return samples

    # The sampling should only happen on exactly one TPU chip of exactly
    # one process. Pass `the One` boolean to chips.
    the_one_mask_global = jnp.arange(0, jax.device_count()) == 0
    assert np.sum(the_one_mask_global).item() == 1
    the_one_mask_pmap = global_to_local_pmap(the_one_mask_global)

    @jax.util.wraps(fn)
    def _result(rng_key):
        rng_key_pmap = utils.bcast_local_devices(rng_key)
        # The sampling returns pmap-ed data.
        samples_pmap = _pmap_run_fn(rng_key_pmap, the_one_mask_pmap)
        chex.assert_tree_shape_prefix(samples_pmap, (jax.local_device_count(),))

        # All chips/devices will return exactly the same data. Thus pick one.
        samples = jax.device_get(utils.get_first(samples_pmap))

        return samples

    return _result


def center_crop(images: chex.Array, bounding) -> chex.Array:
    if images.shape[-3:-1] == bounding:
        return images

    start = tuple(map(lambda a, da: a // 2 - da // 2, images.shape[-3:-1], bounding))
    end = tuple(map(operator.add, start, bounding))
    return images[..., start[0] : end[0], start[1] : end[1], :]


def create_optimizer(config: config_dict.ConfigDict) -> optax.GradientTransformation:
    transformations = []

    for transform_config in config:
        name = transform_config["name"]
        kwargs = transform_config.get("kwargs", dict())
        transform = getattr(optax, name)
        transformations.append(transform(**kwargs))

    return optax.chain(*transformations)


class Identity(hk.Module):
    """Torso for Identity."""

    def __call__(self, x: chex.Array) -> chex.Array:
        return x
