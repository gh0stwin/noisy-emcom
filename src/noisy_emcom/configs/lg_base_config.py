"""Base config."""

import pathlib as pl

from jaxline import base_config
from ml_collections import config_dict

from noisy_emcom.utils import types
from noisy_emcom.utils.config import LazyField, get_value

TASK_OVERRIDE = {}


def get_config(sweep="debug"):
    """Return base config object for training or testing."""
    home_path = str(pl.Path.home())
    data = ""

    if "imagenet" in sweep:
        data = "imagenet"
    elif "celeba" in sweep:
        data = "celeba"

    config = base_config.get_base_config()
    config.experiment_mode = "fit"

    # Define global storage folder (ckpt, logs etc.)
    config.checkpoint_dir = "./.tmp/cidre_ckpts"
    config.checkpoint_interval_type = "steps"
    config.save_checkpoint_interval = int(7.5e5)

    # Basic jaxline logging options
    config.interval_type = "secs"
    config.log_train_data_interval = 30
    config.log_tensors_interval = 30

    # Put here values that are referenced multiple times
    config.training_steps = int(7.5e5)
    config.batch_size = 1024
    config.vocab_size = 20
    config.length = 10
    config.task = types.Task.DISCRIMINATION

    config.n_speakers = 1
    config.n_listeners = 1

    config.listener_core_hidden_size = 512

    config.log_task_name = ""
    config.log_tags = [
        "game:lg",
        "type:rl-rl",
        "noise:0",
        "idk:0",
        "rounds:1",
        "pop:1-1",
        "data:" + data,
    ]

    config.log_lazy_tags = [
        "mode:" + get_value("experiment_mode", config),
        LazyField(lambda c=config: f"bs:{c.batch_size}"),
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
                log_id=get_value("log_id", config),
                training_mode="training",
                training=dict(
                    batch_size=get_value("batch_size", config),
                    length=get_value("length", config),
                    steps=get_value("training_steps", config),
                ),
                population=dict(
                    agents_count=dict(speaker=1, listener=1), num_agents_per_step=1
                ),
                agent=dict(
                    agent_type=types.AgentType.LEWIS_GAME,
                    kwargs=dict(
                        speaker=dict(
                            length=get_value("length", config),
                            vocab_size=get_value("vocab_size", config),
                            sos_token=get_value("vocab_size", config),
                            torso_config=dict(
                                torso_type=types.TorsoType.IDENTITY, torso_kwargs=dict()
                            ),
                            embedder_config=dict(
                                torso_type=types.TorsoType.DISCRETE,
                                torso_kwargs=dict(
                                    vocab_size=get_value("vocab_size", config) + 1,
                                    embed_dim=10,
                                    mlp_kwargs=dict(output_sizes=()),
                                ),
                            ),
                            core_config=dict(
                                core_type=types.CoreType.LSTM,
                                core_kwargs=dict(hidden_size=256),
                            ),
                            head_config=dict(
                                head_type=types.SpeakerHeadType.POLICY_QVALUE_DUELING,
                                head_kwargs=dict(
                                    hidden_sizes=(),
                                    num_actions=get_value("vocab_size", config),
                                ),
                                kwargs=dict(),
                            ),
                        ),
                        listener=dict(
                            listener_type=types.ListenerType.RL,
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
                                    core_kwargs=dict(
                                        hidden_size=get_value(
                                            "listener_core_hidden_size", config
                                        )
                                    ),
                                ),
                                head_config=dict(
                                    head_type=types.ListenerHeadType.NORM_MERGE_POL_VAL,
                                    head_kwargs=dict(
                                        message_hidden_sizes=(256,),
                                        games_hidden_sizes=(256,),
                                        value_act_func=None,
                                        value_net_size=(
                                            LazyField(
                                                lambda c=config: int(
                                                    max(c.batch_size / 4, 4)
                                                )
                                            ),
                                            LazyField(
                                                lambda c=config: int(
                                                    max(c.batch_size / 16, 2)
                                                )
                                            ),
                                            1,
                                        ),
                                    ),
                                    kwargs=dict(),
                                ),
                                task=config.task,
                            ),
                        ),
                        channel=dict(
                            channel_type=types.ChannelType.DEFAULT,
                            channel_kwargs=dict(),
                        ),
                        loss=dict(
                            speaker=dict(
                                loss_type=types.SpeakerLossType.REINFORCE,
                                use_baseline=True,
                                speaker_entropy=1e-4,
                                kwargs=dict(
                                    policy_gradient=dict(),
                                    reinforce=dict(speaker_kl_target=0.5),
                                ),
                            ),
                            listener=dict(
                                loss_type=types.ListenerLossType.REINFORCE,
                                kwargs=dict(
                                    classif=dict(task=get_value("task", config)),
                                    cpc=dict(
                                        reward_type=types.RewardType.SUCCESS_RATE,
                                        num_distractors=-1,
                                        cross_device=True,
                                    ),
                                    reinforce=dict(
                                        reward_config=dict(
                                            reward_type=types.GameRewardType.DEFAULT,
                                            kwargs=dict(
                                                failure=-1,
                                                success=1,
                                            ),
                                        ),
                                        entropy_scheduler="constant_schedule",
                                        entropy_scheduler_kwargs=dict(value=1e-4),
                                        use_baseline=True,
                                        critic_weight=1e-3,
                                    ),
                                ),
                                wrappers=[],
                                wrapper_kwargs=[],
                            ),
                        ),
                        opt=dict(
                            speaker=[
                                dict(name="adam", kwargs=dict(learning_rate=1e-4))
                            ],
                            listener=[
                                dict(name="adam", kwargs=dict(learning_rate=5e-5))
                            ],
                        ),
                        speaker_update_ema=0.99,
                    ),
                ),
                imitation=dict(
                    nbr_students=1,
                    imitation_step=None,
                    imitation_type=types.ImitationMode.BEST,
                    self_imitation=False,
                ),
                reset=dict(
                    reset_step=0,
                    reset_type=types.ResetMode.PAIR,
                    reset_kwargs=dict(
                        n_speakers=get_value("n_speakers", config),
                        n_listeners=get_value("n_listeners", config),
                        reset_opt=False,
                        components_to_replace=["head"],
                    ),
                ),
                reset_opt=dict(
                    reset_mode=types.ResetMode.PAIR, reset_step=LazyField(lambda: 0)
                ),
                evaluation=dict(
                    eval_type="default",
                    eval_topsim_mode=False,
                    all_batch_sizes=(16, 64, 256, 1024, 4096),
                    batch_size=16,
                    subsampling_ratio=0.2,
                    subsampling_ratios=(0.2, 0.2, 0.2, 0.1, 0.02),
                    max_n_agents=10,
                    topsim=dict(
                        classif=dict(
                            meaning=types.MeaningSimilarity.INPUTS,
                            task=types.Task.CLASSIFICATION,
                        )
                    ),
                    channel_type=types.ChannelType.DETECTABLE,
                ),
                game=dict(
                    name="logit",
                    kwargs=dict(
                        dummy=dict(  # Dataset used for testing.
                            max_steps=get_value("training_steps", config)
                        ),
                        logit=dict(
                            dataset_name="byol_imagenet2012",
                            dataset_path=f"{home_path}/tensorflow_datasets",
                            num_eval_epochs=5,
                            shuffle_evaluation=True,
                            shuffle_training=True,
                            divide_candidates_into_devices=False,
                            coeff_noise=0.0,
                            has_noise=False,
                            is_one_hot_label=False,
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
                        filename="agents",
                    ),
                    restore_path="",
                ),
            ),
        )
    )

    config.checkpointer = config_dict.ConfigDict(dict(type="none", kwargs=dict()))
    exp_config = config.experiment_kwargs.config

    if sweep == "debug":
        config.experiment_kwargs.config.debug = True
        config.training_steps = 1
        config.interval_type = "steps"
        config.log_train_data_interval = 1
        config.log_tensors_interval = 1
        exp_config.checkpointing.save_checkpoint_interval = 1
        exp_config.training.batch_size = 8
        exp_config.evaluation.batch_size = 8
        exp_config.evaluation.subsampling_ratio = 0.5
    elif sweep in ("celeba", "celeba_noimg", "celeba_logits"):
        if sweep == "celeba":
            exp_config.game.kwargs.logit.dataset_name = "byol_celeb_a2"
        elif sweep == "celeba_noimg":
            exp_config.game.kwargs.logit.dataset_name = "byol_celeb_a2_noimg"
        elif sweep == "celeba_logits":
            exp_config.game.kwargs.logit.dataset_name = "byol_celeb_a2_logits"

        # exp_config.evaluation.topsim.attrs = dict(
        #     meaning=types.MeaningSimilarity.ATTRIBUTES, task=types.Task.ATTRIBUTE
        # )
    elif sweep == "imagenet":
        pass
    elif sweep == "imagenet_modular":
        config.experiment_kwargs.config.game.name = "modular"
    elif sweep == "multiple":
        config.experiment_kwargs.config.game.name = "multiple"
    elif sweep == "imagenet_imitation":
        # Set population size
        exp_config.population.n_speakers = 10
        exp_config.population.n_listeners = 10
        exp_config.population.num_agents_per_step = 10
        # Set imitation parameters
        exp_config.imitation.nbr_students = 4
        exp_config.imitation.imitation_step = 10
    else:
        raise ValueError(f"Sweep {sweep} not recognized.")

    return config
