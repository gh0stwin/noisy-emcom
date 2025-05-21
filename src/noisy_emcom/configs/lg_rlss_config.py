"""Base config."""

from noisy_emcom.configs import lg_base_config
from noisy_emcom.utils import types

TASK_OVERRIDE = {}


def get_config(sweep="debug"):
    """Return config object for training or testing."""

    config = lg_base_config.get_config(sweep)

    config.experiment = "lg_rlss"
    agent_config = config.experiment_kwargs.config.agent
    agent_config.kwargs.listener.listener_type = types.ListenerType.SS
    agent_config.kwargs.listener.kwargs.head_config = dict(
        head_type=types.ListenerHeadType.CPC,
        head_kwargs=dict(hidden_sizes=(256,)),
        kwargs=dict(),
    )

    agent_config.kwargs.loss.listener.loss_type = types.ListenerLossType.CPC

    tags = config.log_tags
    tags = [tag for tag in tags if "type:" not in tag] + ["type:rl-ss"]
    config.log_tags = tags

    # Prevents accidentally setting keys that aren't recognized (e.g. in tests).
    config.lock()
    return config
