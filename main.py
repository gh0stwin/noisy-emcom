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

"""Specifies which experiment to launch."""

import sys

sys.path.append("./src/noisy_emcom/")

from typing import Any

from absl import app, flags
import jax
from jax.lib import xla_bridge
from jaxline import platform
from jaxline.experiment import AbstractExperiment
from ml_collections import config_dict

from noisy_emcom.experiments import eol_experiment, lewis_experiment
from noisy_emcom.utils import eval_utils
import noisy_emcom.utils.config as emcom_config

FLAGS = flags.FLAGS


def main(argv):
    flags.mark_flag_as_required("config")
    config = FLAGS.config
    config = emcom_config.resolve_dictionary(config)
    FLAGS.config = config

    if config.experiment in ("lg_rlss", "lg_rlrl", "nlg_rlrl"):
        if config.experiment_mode == "fit":
            _fit(lewis_experiment.LewisExperiment, argv)
        elif config.experiment_mode == "testchannel":
            # Deterministic eval
            _test(config, mode="eval_test_average")
        elif config.experiment_mode == "testinput":
            _test_inputs(config, mode="eval_test_average")
        elif config.experiment_mode == "testmessage":
            _test_messages(config, mode="eval_test_average")
        elif config.experiment_mode == "testtopsim":
            _test_topsim(config, mode="eval_test_average")
    elif config.experiment == "etl":
        if config.experiment_mode == "fit":
            _fit(eol_experiment.EaseOfLearningExperiment, argv)
        elif config.experiment_mode == "test":
            eval_utils.evaluate_etl(config, mode="eval_test_average")
    else:
        raise ValueError(
            f"{config.experiment} not recognized. "
            "Only lewis and ease_of_learning are supported"
        )


def _jax_devices():
    print()
    print(f"jax backend: {xla_bridge.get_backend().platform}")
    print("jax devices:")
    print(jax.devices())
    print(jax.local_devices())
    print(end="\n\n")


def _fit(experiment: AbstractExperiment, argv: Any):
    platform.main(experiment, argv)


def _test(config: config_dict.ConfigDict, mode: str):
    eval_utils.evaluate_final(config, mode)


def _test_inputs(config: config_dict.ConfigDict, mode: str):
    eval_utils.evaluate_noisy_inputs(config, mode)


def _test_messages(config: config_dict.ConfigDict, mode: str):
    eval_utils.evaluate_mess_struct(config, mode)


def _test_topsim(config: config_dict.ConfigDict, mode: str):
    assert config.experiment_kwargs.config.evaluation.eval_topsim_mode == True
    eval_utils.evaluate_final(config, mode)


if __name__ == "__main__":
    _jax_devices()
    app.run(main)
