"""Base config."""

from noisy_emcom.configs import lg_base_config
from noisy_emcom.utils import types

TASK_OVERRIDE = {}


def get_config(sweep="debug"):
    """Return config object for training or testing."""

    config = lg_base_config.get_config(sweep)

    config.experiment = "lg_rlrl"
    listener_kwargs = config.experiment_kwargs.config.agent.kwargs.listener.kwargs
    listener_kwargs.policy_actuator_config = dict(
        policy_actuator_type=types.PolicyActuatorType.DEFAULT,
        kwargs=dict(),
    )

    listener_kwargs.temp_scheduler = "constant_schedule"
    listener_kwargs.temp_scheduler_kwargs = dict(value=1.0)

    # Prevents accidentally setting keys that aren't recognized (e.g. in tests).
    config.lock()
    return config
