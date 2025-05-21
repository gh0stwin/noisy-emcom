"""Config related util functions."""
from typing import Any, Callable

import clearml
import jax
from ml_collections import config_dict

from noisy_emcom.utils import types


class LazyField:
    def __init__(self, caller: Callable) -> None:
        self._caller = caller

    def resolve(self) -> Any:
        return self._caller()


def define_experiment_name(config: config_dict.ConfigDict) -> str:
    if "test" in config.experiment_mode:
        return _test_experiment_name(config)

    if config.experiment == "etl":
        return _etl_experiment_name(config)

    if "lg" in config.experiment:
        return _lg_experiment_name(config)

    raise ValueError(
        f"Unknown 'config.experiment_name': {config.experiment};"
        + f" or 'config.experimnent_mode': {config.experiment_mode}"
    )


def get_api_token(file_path):
    "read api token from file."
    with open(file_path, "r", encoding="utf-8") as file:
        api_token = file.read()

    return api_token


def resolve_dictionary(
    config: config_dict.ConfigDict,
) -> config_dict.ConfigDict:
    """Resolve the 'LazyField(...) / get_value(...) / get_oneway_ref(...)' in the
    dictionnary.

    """
    static_config = config
    static_config = config_dict.ConfigDict(
        jax.tree_util.tree_map(_resolve_field, static_config.to_dict())
    )

    static_config = config_dict.ConfigDict(
        jax.tree_util.tree_map(_resolve_reference, static_config.to_dict())
    )

    return static_config


def get_value(key, config):
    return config.get_oneway_ref(key)


def _resolve_field(entry):
    if isinstance(entry, LazyField):
        return entry.resolve()

    return entry


def _resolve_reference(entry):
    if isinstance(entry, config_dict.FieldReference):
        return entry.get()

    return entry


def _etl_experiment_name(config: config_dict.ConfigDict) -> str:
    task_name = config.experiment
    project_name = config.logger.kwargs.project_name
    noise = "0.0"
    channel_config = config.experiment_kwargs.config.agent.kwargs.channel

    if channel_config.channel_type == types.ChannelType.DETECTABLE:
        noise = str(channel_config.channel_kwargs.scheduler_kwargs.value)

    model_info = config.experiment_kwargs.config.checkpoint_experiment.kwargs
    model_info = model_info.load_model_info
    model = clearml.InputModel.query_models(**model_info.default)[0]
    lg_task = clearml.Task.get_task(task_id=model.task)
    lg_task.close()

    task_regex = (
        rf"^{task_name}_{config.task}_{noise}_\+?(\d+(\.(\d+)?)?|\.\d+)_{lg_task.name}$"
    )

    tasks = clearml.Task.get_tasks(project_name=project_name, task_name=task_regex)
    idx = len(tasks)

    for task in tasks:
        task.close()

    task_name = f"{task_name}_{config.task}_{noise}_{idx}_{lg_task.name}"
    return task_name


def _lg_experiment_name(config: config_dict.ConfigDict) -> str:
    task_name = config.experiment
    project_name = config.logger.kwargs.project_name
    noise = "0.0"
    dataset_name = [tag for tag in config.log_tags if "data:" in tag][0]
    dataset_name = dataset_name.rsplit(":", 1)[-1]

    if task_name.startswith("nlg"):
        channel_config = config.experiment_kwargs.config.agent.kwargs.channel
        noise = str(channel_config.channel_kwargs.scheduler_kwargs.end_value)

    task_name = f"{task_name}_{noise}_{config.batch_size}_{dataset_name}"
    task_regex = rf"^{task_name}_\+?(\d+(\.(\d+)?)?|\.\d+)$"
    tasks = clearml.Task.get_tasks(project_name=project_name, task_name=task_regex)
    task_idxs = set()
    expected_task_idxs = set(range(len(tasks) + 1))

    for task in tasks:
        task_idxs.add(int(task.name.rsplit("_", 1)[-1]))
        task.close()

    idx = sorted(list(expected_task_idxs - task_idxs))[0]
    task_name = f"{task_name}_{idx}"
    return task_name


def _test_experiment_name(config: config_dict.ConfigDict) -> str:
    model_info = config.experiment_kwargs.config.checkpointing.kwargs.load_model_info
    model = clearml.InputModel.query_models(**model_info.default)[0]
    task = clearml.Task.get_task(task_id=model.task)
    task_name = task.name
    task_config = task.get_configuration_object_as_dict("config")
    task_config = config_dict.ConfigDict(task_config)
    task.close()
    idx = 0  # For now tests are deterministic

    # Assert LG configs
    if task_name.startswith("nlg"):
        channel_config = config.experiment_kwargs.config.agent.kwargs.channel
        channel = channel_config.channel_kwargs.scheduler_kwargs.end_value
        task_channel_config = task_config.experiment_kwargs.config.agent.kwargs.channel
        task_channel = task_channel_config.channel_kwargs.scheduler_kwargs.end_value
        assert channel == task_channel

    return f"{config.experiment_mode}_{idx}_{task_name}"
