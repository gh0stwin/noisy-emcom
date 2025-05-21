"""Base config."""

from noisy_emcom.configs import lg_rlrl_config
from noisy_emcom.utils import types
from noisy_emcom.utils.config import LazyField, get_value

TASK_OVERRIDE = {}


def get_config(sweep="debug_0.25"):
    """Return config object for training or testing."""

    base_sweep, channel_sweep = sweep.rsplit("_", 1)
    config = lg_rlrl_config.get_config(base_sweep)
    config.unlock()

    channel_sweep = float(channel_sweep)

    config.experiment = "nlg_rlrl"
    listener_kwargs = config.experiment_kwargs.config.agent.kwargs.listener.kwargs
    listener_kwargs.torso_config.torso_kwargs.vocab_size = (
        get_value("vocab_size", config) + 1
    )

    channel = config.experiment_kwargs.config.agent.kwargs.channel
    channel.channel_kwargs = dict(
        scheduler="linear_schedule",
        scheduler_kwargs=dict(
            init_value=0,
            end_value=channel_sweep,
            transition_steps=LazyField(lambda c=config: int(0.4 * c.training_steps)),
        ),
    )

    channel.channel_type = types.ChannelType.DETECTABLE
    channel.channel_kwargs.unknown_token = get_value("vocab_size", config)

    tags = config.log_tags
    tags = [tag for tag in tags if "game:" not in tag] + ["game:nlg"]
    tags = [tag for tag in tags if "noise:" not in tag] + [f"noise:{channel_sweep}"]
    config.log_tags = tags

    # Prevents accidentally setting keys that aren't recognized (e.g. in tests).
    config.lock()
    return config
