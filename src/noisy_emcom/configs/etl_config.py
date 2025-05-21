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

"""Base config."""

import pathlib as pl

from jaxline import base_config
from ml_collections import config_dict

from noisy_emcom.utils import types
from noisy_emcom.utils.config import LazyField, get_value

TASK_OVERRIDE = {}


def get_config(sweep="debug_classification_0"):
    """Return config object for training."""
    data_sweep, task_sweep, noise_sweep = sweep.rsplit("_", 2)
    data = ""

    if "imagenet" in sweep:
        data = "imagenet"
    elif "celeba" in sweep:
        data = "celeba"

    config = base_config.get_base_config()

    def _get_task(config=config):
        task = config.task

        if "discr" in task:
            task = f"{task}-{config.batch_size}"

        return f"task:{task}"

    config.experiment = "etl"
    config.experiment_mode = "fit"

    # Define global storage folder (ckpt, logs etc.)
    config.checkpoint_dir = "./.tmp/cidre_ckpts"
    config.checkpoint_interval_type = "steps"
    config.save_checkpoint_interval = int(1e4)

    # Basic jaxline logging options
    config.interval_type = "secs"
    config.log_train_data_interval = 60
    config.log_tensors_interval = 60

    config.training_steps = int(1e4)

    # Put here values that are referenced multiple times
    config.batch_size = 128
    config.vocab_size = 20
    config.length = 10
    config.task = task_sweep

    config.log_task_name = ""
    config.log_tags = [
        "game:etl",
        f"noise:{noise_sweep}",
        f"data:{data}",
    ]

    config.log_lazy_tags = [
        "mode:" + get_value("experiment_mode", config),
        LazyField(lambda c=config: f"bs:{c.batch_size}"),
        LazyField(_get_task),
    ]

    config.logger = config_dict.ConfigDict(
        dict(
            type="local",
            kwargs=dict(
                project_name="gh0stwin/noisy-emcom",
                task_name=get_value("log_task_name", config),
                task_type="training",
                tags=get_value("log_tags", config) + get_value("log_lazy_tags", config),
            ),
        )
    )

    config.experiment_kwargs = config_dict.ConfigDict(
        dict(
            config=dict(
                debug=False,
                training_mode="eval",
                speaker_index=0,
                task=task_sweep,
                training=dict(
                    batch_size=get_value("batch_size", config),
                    length=get_value("length", config),
                    steps=get_value("training_steps", config),
                    training_mode=types.TrainingMode.EVAL,
                ),
                agent=dict(
                    agent_type=types.AgentType.LEWIS_GAME,
                    kwargs=dict(
                        speaker=dict(),
                        listener=dict(
                            listener_type=types.ListenerType.SS,
                            kwargs=dict(
                                torso_config=dict(
                                    torso_type=types.TorsoType.DISCRETE,
                                    torso_kwargs=dict(
                                        vocab_size=get_value("vocab_size", config),
                                        embed_dim=10,
                                        mlp_kwargs=dict(output_sizes=()),
                                    ),
                                ),
                                core_config=dict(
                                    core_type=types.CoreType.STATIC_RNN,
                                    core_kwargs=dict(hidden_size=512),
                                ),
                                head_config=dict(
                                    head_type=types.ListenerHeadType.MULTIMLP,
                                    head_kwargs=dict(
                                        hidden_sizes=(256,), task=task_sweep
                                    ),
                                    kwargs=dict(),
                                ),
                                task=task_sweep,
                            ),
                        ),
                        channel=dict(
                            channel_type=types.ChannelType.DEFAULT,
                            channel_kwargs=dict(),
                        ),
                        loss=dict(
                            speaker=dict(),
                            listener=dict(
                                loss_type=types.ListenerLossType.CLASSIF,
                                kwargs=dict(
                                    classif=dict(
                                        reward_type=types.RewardType.SUCCESS_RATE,
                                        task=get_value("task", config),
                                    ),
                                    cpc=dict(
                                        reward_type=types.RewardType.SUCCESS_RATE,
                                        num_distractors=-1,
                                        cross_device=False,
                                    ),
                                    reconstruction=dict(
                                        reward_type=types.RewardType.LOG_PROB
                                    ),
                                ),
                            ),
                        ),
                        opt=dict(
                            speaker=[dict(name="sgd", kwargs=dict(learning_rate=0.0))],
                            listener=[
                                dict(name="adam", kwargs=dict(learning_rate=1e-3))
                            ],
                        ),
                        speaker_update_ema=0.0,
                    ),
                ),
                population=dict(
                    agents_count=dict(speaker=1, listener=1), num_agents_per_step=1
                ),  # Unused for EOL
                imitation=dict(),  # Unused for EOL
                reset=dict(),  # Unused for EOL
                reset_opt=dict(),  # Unesed for EOL
                evaluation=dict(
                    eval_type="default", batch_size=get_value("batch_size", config)
                ),
                game=dict(
                    name="logit",
                    kwargs=dict(
                        dummy=dict(max_steps=get_value("training_steps", config)),
                        logit=dict(
                            dataset_name="byol_imagenet2012",
                            dataset_path=f"{str(pl.Path.home())}/tensorflow_datasets/",
                            coeff_noise=0.0,
                            shuffle_training=True,
                            shuffle_evaluation=True,
                            divide_candidates_into_devices=False,
                            is_one_hot_label=True,
                            drop_remainder=True,
                        ),
                    ),
                ),
                checkpointing=dict(
                    type=types.CheckpointerType.LOCAL,
                    kwargs=dict(
                        use_checkpointing=True,
                        checkpoint_dir=get_value("checkpoint_dir", config),
                        save_checkpoint_interval=get_value(
                            "save_checkpoint_interval", config
                        ),
                        filename="etl",
                    ),
                    restore_path="",
                ),
                checkpoint_experiment=dict(
                    type=types.CheckpointerType.LOCAL,
                    kwargs=dict(
                        use_checkpointing=True,
                        checkpoint_dir=get_value("checkpoint_dir", config),
                        save_checkpoint_interval=0,
                        filename="agents",
                    ),
                    restore_path="",
                ),
            ),
        )
    )

    config.checkpointer = config_dict.ConfigDict(dict(type="none", kwargs=dict()))
    exp_config = config.experiment_kwargs.config

    if data_sweep == "debug":
        config.experiment_kwargs.config.debug = True
        config.interval_type = "steps"
        config.training_steps = int(1)
        config.log_train_data_sweep_interval = 1
        config.log_tensors_interval = 1
        exp_config.training.batch_size = 8
    elif data_sweep == "celeba":
        exp_config.game.kwargs.logit.dataset_name = "byol_celeb_a2"
    elif data_sweep == "celeba_logits":
        exp_config.game.kwargs.logit.dataset_name = "byol_celeb_a2_logits"
    elif data_sweep == "celeba_noimg":
        exp_config.game.kwargs.logit.dataset_name = "byol_celeb_a2_noimg"
    elif data_sweep == "imagenet":
        pass
    else:
        raise ValueError(f"data_sweep with value '{data_sweep}' is not recognized.")

    if task_sweep == types.Task.ATTRIBUTE:
        pass
    elif task_sweep == types.Task.CLASSIFICATION:
        exp_config.agent.kwargs.loss.listener.kwargs.classif.rank_acc = 5
    elif task_sweep == types.Task.DISCRIMINATION:
        exp_config.game.kwargs.logit.has_noise = True
        exp_config.game.kwargs.logit.coeff_noise = 0.5
        exp_config.agent.kwargs.listener.kwargs.head_config = dict(
            head_type=types.ListenerHeadType.DISCRIMINATION,
            head_kwargs=dict(hidden_sizes=(256,), activation_func="tanh"),
            kwargs=dict(),
        )
    elif task_sweep == types.Task.DISCRIMINATION_DIST:
        exp_config.game.kwargs.logit.has_noise = True
        exp_config.game.kwargs.logit.coeff_noise = 0.5
        exp_config.agent.kwargs.loss.listener.loss_type = types.ListenerLossType.CPC
        exp_config.agent.kwargs.listener.kwargs.head_config = dict(
            head_type=types.ListenerHeadType.DISCRIMINATION_DIST,
            head_kwargs=dict(hidden_sizes=(256,), activation_func="tanh"),
            kwargs=dict(),
        )
    elif task_sweep == types.Task.RECONSTRUCTION:
        exp_config.agent.kwargs.loss.listener.loss_type = (
            types.ListenerLossType.RECONSTRUCTION
        )

        exp_config.agent.kwargs.opt.listener = [
            dict(name="clip_by_global_norm", kwargs=dict(max_norm=500)),
            dict(
                name="adamw",
                kwargs=dict(learning_rate=3e-4, b1=0.9, b2=0.9, weight_decay=0.01),
            ),
        ]

        exp_config.agent.kwargs.listener.kwargs.head_config = dict(
            head_type=types.ListenerHeadType.RECONSTRUCTION,
            head_kwargs=dict(latent_size=(4, 4, 128), upsample_factor=2),
            kwargs=dict(),
        )
    else:
        raise ValueError(f"task_sweep with value '{task_sweep}' is not recognized.")

    noise_sweep = float(noise_sweep)

    if noise_sweep != 0:
        channel = config.experiment_kwargs.config.agent.kwargs.channel
        channel.channel_type = types.ChannelType.DETECTABLE
        channel.channel_kwargs = dict(
            unknown_token=get_value("vocab_size", config),
            scheduler="constant_schedule",
            scheduler_kwargs=dict(value=noise_sweep),
        )

    # Prevents accidentally setting keys that aren't recognized (e.g. in tests).
    config.lock()
    return config
