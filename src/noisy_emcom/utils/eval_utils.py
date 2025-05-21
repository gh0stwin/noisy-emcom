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

import copy
import itertools
from itertools import product
import math
import random
from typing import Optional, Text

from absl import flags, logging
import chex
import jax
from jaxline.platform import create_writer
from jaxline.writers import Writer
from ml_collections import config_dict
import numpy as np

from noisy_emcom.experiments.eol_experiment import EaseOfLearningExperiment
from noisy_emcom.experiments.lewis_experiment import LewisExperiment
from noisy_emcom.utils import types

FLAGS = flags.FLAGS


def evaluate_final(
    config: Optional[config_dict.ConfigDict], mode: Optional[Text]
) -> None:
    """The main evaluation loop.

    This loop loads a checkpoint and evaluates its performance on the
    test set, by calling experiment.evaluate.

    Args:
      config: Optional argument. Defines the config.
      mode: optional argument. Defines the mode of evalution. Coud be any value in
      eval_{test/valid}_{average/vote}. Default (eval_test_average).
      rng: select evaluation seed (recommended to always use the same)

    """
    if not config.experiment_kwargs.config.checkpointing.kwargs.use_checkpointing:
        logging.info("\nCheckpointing not available for evaluation.\n")
        return

    lewis_exp_cls = LewisExperiment
    writer = create_writer(config, "eval")
    logging.info("\nEvaluating the final checkpoint on the test set.\n")
    default_rng = jax.random.PRNGKey(42)
    init_rng, eval_rng = jax.random.split(default_rng)
    ckpt_rm_keys = []
    base_config = config.experiment_kwargs.config
    batch_sizes = base_config.evaluation.get(
        "all_batch_sizes", [base_config.evaluation.batch_size]
    )

    for i, batch in enumerate(batch_sizes):
        test_mode = "evallg"
        run_config = copy.deepcopy(base_config)
        run_config.training_mode = test_mode
        run_config.evaluation.batch_size = batch
        run_config.evaluation.subsampling_ratio = (
            base_config.evaluation.subsampling_ratios[i]
        )

        run_config = _add_default_channel(config, run_config)
        exp = lewis_exp_cls(mode=mode, init_rng=init_rng, config=run_config)
        _run_and_log_eval(
            exp,
            eval_rng,
            writer,
            test_mode,
            0,
            str(batch),
            0,
            ckpt_rm_keys,
            restore_path=run_config.checkpointing.restore_path,
        )

    runs = base_config.evaluation.get("ilg_runs", 1)
    # noises = base_config.evaluation.get("ilg_noises", (0.25, 0.5, 0.75))
    noises = (0.25, 0.5, 0.75)

    for i, (run, batch, noise) in enumerate(product(range(runs), batch_sizes, noises)):
        batch_idx = batch_sizes.index(batch)
        test_mode = "evalilg"
        run_config = copy.deepcopy(base_config)
        run_config.training_mode = test_mode
        run_config.evaluation.batch_size = batch
        run_config.evaluation.subsampling_ratio = (
            base_config.evaluation.subsampling_ratios[batch_idx]
        )

        run_config = _add_noise_channel(config, run_config, noise)
        exp = lewis_exp_cls(mode=mode, init_rng=init_rng, config=run_config)
        _run_and_log_eval(
            exp,
            eval_rng,
            writer,
            test_mode,
            noise,
            str(batch),
            run,
            ckpt_rm_keys,
            restore_path=run_config.checkpointing.restore_path,
        )


def evaluate_noisy_inputs(
    config: Optional[config_dict.ConfigDict], mode: Optional[Text]
) -> None:
    if not config.experiment_kwargs.config.checkpointing.kwargs.use_checkpointing:
        logging.info("\nCheckpointing not available for evaluation.\n")
        return

    lewis_exp_cls = LewisExperiment
    writer = create_writer(config, "eval")
    logging.info("\nEvaluating the final checkpoint on the test set.\n")
    default_rng = jax.random.PRNGKey(42)
    init_rng, eval_rng = jax.random.split(default_rng)
    base_config = config.experiment_kwargs.config
    ckpt_rm_keys = []
    batch_sizes = base_config.evaluation.get(
        "all_batch_sizes", [base_config.evaluation.batch_size]
    )

    for i, batch in enumerate(batch_sizes):
        test_mode = "evallg"
        run_config = copy.deepcopy(base_config)
        run_config.training_mode = test_mode
        run_config.evaluation.batch_size = batch
        run_config.evaluation.subsampling_ratio = (
            base_config.evaluation.subsampling_ratios[i]
        )

        run_config = _add_default_channel(config, run_config)
        exp = lewis_exp_cls(mode=mode, init_rng=init_rng, config=run_config)
        _run_and_log_eval(
            exp,
            eval_rng,
            writer,
            test_mode,
            0,
            str(batch),
            0,
            ckpt_rm_keys,
            restore_path=run_config.checkpointing.restore_path,
        )

    runs = base_config.evaluation.get("ilg_runs", 1)
    # noises = base_config.evaluation.get("ilg_noises", (0.25, 0.5, 0.75))
    noises = (0.25, 0.5, 0.75)

    for i, (run, batch, noise) in enumerate(product(range(runs), batch_sizes, noises)):
        batch_idx = batch_sizes.index(batch)
        test_mode = "evalilg"
        run_config = copy.deepcopy(base_config)
        run_config.training_mode = test_mode
        run_config.evaluation.batch_size = batch
        run_config.evaluation.subsampling_ratio = (
            base_config.evaluation.subsampling_ratios[batch_idx]
        )

        run_config = _add_noise_channel(config, run_config, noise)
        exp = lewis_exp_cls(mode=mode, init_rng=init_rng, config=run_config)
        _run_and_log_eval(
            exp,
            eval_rng,
            writer,
            test_mode,
            noise,
            str(batch),
            run,
            ckpt_rm_keys,
            restore_path=run_config.checkpointing.restore_path,
        )


def evaluate_noisy_inputs_noisy_channel(
    config: Optional[config_dict.ConfigDict],
    mode: Optional[Text],
    noise_levels: list[float],
) -> None:
    if not config.experiment_kwargs.config.checkpointing.kwargs.use_checkpointing:
        logging.info("\nCheckpointing not available for evaluation.\n")
        return

    lewis_exp_cls = LewisExperiment
    writer = create_writer(config, "eval")
    logging.info("\nEvaluating the final checkpoint on the test set.\n")
    default_rng = jax.random.PRNGKey(42)
    init_rng, eval_rng = jax.random.split(default_rng)
    base_config = config.experiment_kwargs.config
    ckpt_rm_keys = []
    batch_sizes = base_config.evaluation.get(
        "all_batch_sizes", [base_config.evaluation.batch_size]
    )

    for i, batch in enumerate(batch_sizes):
        test_mode = "evalilg"
        run_config = copy.deepcopy(base_config)
        run_config.training_mode = test_mode
        run_config.evaluation.batch_size = batch
        run_config.evaluation.subsampling_ratio = (
            base_config.evaluation.subsampling_ratios[i]
        )

        for noise in noise_levels:
            run_config = _add_noise_channel(config, run_config, noise)
            exp = lewis_exp_cls(mode=mode, init_rng=init_rng, config=run_config)
            _run_and_log_eval(
                exp,
                eval_rng,
                writer,
                test_mode,
                noise,
                str(batch),
                0,
                ckpt_rm_keys,
                restore_path=run_config.checkpointing.restore_path,
            )


def evaluate_mess_struct(
    config: Optional[config_dict.ConfigDict], mode: Optional[Text]
) -> None:
    """The main evaluation loop.

    This loop loads a checkpoint and evaluates its performance on the
    test set, by calling experiment.evaluate.

    Args:
      config: Optional argument. Defines the config.
      mode: optional argument. Defines the mode of evalution. Coud be any value in
      eval_{test/valid}_{average/vote}. Default (eval_test_average).
      rng: select evaluation seed (recommended to always use the same)

    """
    if not config.experiment_kwargs.config.checkpointing.kwargs.use_checkpointing:
        logging.info("\nCheckpointing not available for evaluation.\n")
        return

    lewis_exp_cls = LewisExperiment
    writer = create_writer(config, "eval")
    logging.info("\nEvaluating the final checkpoint on the test set.\n")
    default_rng = jax.random.PRNGKey(42)
    random.seed(42)
    init_rng, eval_rng = jax.random.split(default_rng)
    ckpt_rm_keys = []
    base_config = config.experiment_kwargs.config
    batch_sizes = base_config.evaluation.get(
        "all_batch_sizes", [base_config.evaluation.batch_size]
    )

    start_mask, end_mask = base_config.evaluation.get("mask_tokens_range", (0, 6))
    n_combinations = base_config.evaluation.get("n_mask_combinations", 10)
    combinations = {
        m: _n_random_combinations(range(config.length), m, n_combinations)
        for m in range(start_mask, end_mask)
    }

    for batch in batch_sizes:
        for n_mask in range(start_mask, end_mask):
            mask_combinations = combinations[n_mask]

            for i, comb in enumerate(mask_combinations):
                test_mode = "evallg"
                run_config = copy.deepcopy(base_config)
                run_config.training_mode = test_mode
                run_config.evaluation.batch_size = batch
                run_config = _add_fixed_token_noise_channel(config, run_config, comb)
                exp = lewis_exp_cls(mode=mode, init_rng=init_rng, config=run_config)
                _run_and_log_eval(
                    exp,
                    eval_rng,
                    writer,
                    test_mode,
                    0,
                    f"nr_mask:{n_mask}_batch:{batch}",
                    0,
                    ckpt_rm_keys,
                    restore_path=run_config.checkpointing.restore_path,
                    log_step=i,
                )


def evaluate_etl(
    config: Optional[config_dict.ConfigDict], mode: Optional[Text]
) -> None:
    if not config.experiment_kwargs.config.checkpointing.kwargs.use_checkpointing:
        logging.info("\nCheckpointing not available for evaluation.\n")
        return

    etl_cls = EaseOfLearningExperiment
    writer = create_writer(config, "eval")
    logging.info("\nEvaluating the final checkpoint on the test set.\n")
    rng = jax.random.PRNGKey(config.random_seed)
    exp = etl_cls(mode=mode, init_rng=rng, config=config.experiment_kwargs.config)

    step, _ = exp.restore_state(config.experiment_kwargs.config.checkpoint_experiment.restore_path)
    stats, info = exp.evaluate(global_step=np.array(step), rng=rng, writer=writer)
    images = info.pop("images", None)

    writer.write_scalars(0, stats)

    if images:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        for key, images_array in images.items():
            images_array = np.clip(images_array * std + mean, 0, 1)
            writer.mode = key
            writer.write_images(0, {"face": images_array})


def _add_default_channel(
    base_config: types.Config, exp_config: types.Config
) -> types.Config:
    exp_config.agent.kwargs.channel = dict(
        channel_type=types.ChannelType.DEFAULT, channel_kwargs=dict()
    )

    return exp_config


def _add_noise_channel(
    base_config: types.Config, exp_config: types.Config, noise: float
) -> types.Config:
    exp_config.agent.kwargs.channel = dict(
        channel_type=base_config.experiment_kwargs.config.evaluation.get(
            "channel_type", types.ChannelType.DETECTABLE
        ),
        channel_kwargs=dict(
            unknown_token=base_config.vocab_size,
            scheduler="constant_schedule",
            scheduler_kwargs=dict(value=noise),
        ),
    )

    return exp_config


def _add_fixed_token_noise_channel(
    base_config: types.Config, exp_config: types.Config, combination
):
    exp_config.agent.kwargs.channel = dict(
        channel_type=types.ChannelType.DETECTABLE_FIXED_TOKENS,
        channel_kwargs=dict(
            unknown_token=base_config.vocab_size,
            message_length=base_config.length,
            mask_tokens=combination,
        ),
    )

    return exp_config


def _n_random_combinations(iterable, comb, n):
    combinations = []
    all_combs = math.comb(len(iterable), comb)

    if all_combs <= n:
        return list(itertools.combinations(iterable, comb))

    while len(combinations) < n:
        aux = _random_combination(iterable, comb)

        if aux not in combinations:
            combinations.append(aux)

    return combinations


def _random_combination(iterable, comb):
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), comb))
    return tuple(pool[i] for i in indices)


def _run_and_log_eval(
    exp: LewisExperiment,
    rng: chex.PRNGKey,
    writer: Writer,
    test_mode: str,
    noise: float,
    batch: str,
    run: int,
    ckpt_rm_keys: list,
    restore_path: Optional[str] = None,
    log_step: int = 0,
) -> None:
    mode = f"{test_mode}-{noise}-{batch}-{run}"
    step, _ = exp.restore_state(restore_path, rm_keys=ckpt_rm_keys)
    logging.info("Eval [Step %d] {test mode}-{noise}-{batch}-{run}: %s", step, mode)
    scalars = exp.evaluate(global_step=np.array(step), rng=rng)
    scalars = {f"{key}-{batch}": v for key, v in scalars.items()}
    writer.mode = mode
    writer.write_scalars(log_step, scalars)
