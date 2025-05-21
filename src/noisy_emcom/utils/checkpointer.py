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

"""Generic checkpointer to load and store data."""
import abc
import collections
import datetime
import hashlib
import os
import pickle
import random
from typing import Optional, Union

from absl import logging
import jax
import jax.numpy as jnp
from jaxline import utils
from ml_collections import config_dict

from noisy_emcom.utils import types

CkptData = collections.namedtuple(
    "CkptData", ["experiment_state", "config", "step", "rng"]
)


class Checkpointer(abc.ABC):
    """A checkpoint saving and loading class."""

    def __init__(
        self,
        use_checkpointing: bool,
        checkpoint_dir: str,
        save_checkpoint_interval: int,
        filename: str,
    ):
        if (
            not use_checkpointing
            or checkpoint_dir is None
            # or save_checkpoint_interval <= 0
        ):
            self._checkpoint_enabled = False
            return

        self._checkpoint_enabled = True
        self._last_checkpoint_step = 0
        self._checkpoint_every = save_checkpoint_interval
        self._suffix = ".pkl"
        self._checkpoint_dir = checkpoint_dir
        os.makedirs(self._checkpoint_dir, exist_ok=True)
        self._filename = filename
        self._checkpoint_path = os.path.join(
            self._checkpoint_dir, filename + self._create_unique_hash()
        )

    @property
    def checkpoint_path(self) -> str:
        return self._checkpoint_path + self._suffix

    @abc.abstractmethod
    def load_checkpoint(self, checkpoint_path: Optional[str] = None):
        """Load checkpoint."""

    @abc.abstractmethod
    def _save_checkpoint(self, checkpoint_data, step):
        """Save checkpoint based on subclass."""

    def close(self):
        return

    def maybe_save_checkpoint(
        self,
        xp_state: types.AllProperties,
        config: config_dict.ConfigDict,
        step: int,
        rng: jnp.ndarray,
        is_final: bool,
    ):
        """Saves a checkpoint if enough time has passed since the previous one."""

        # Checks whether we should perform checkpointing.
        if (
            not self._checkpoint_enabled
            or jax.process_index() != 0
            or (  # Only checkpoint the first worker.
                not is_final
                and step - self._last_checkpoint_step < self._checkpoint_every
            )
        ):
            return

        # Creates data to checkpoint.
        checkpoint_data = dict(
            experiment_state=jax.tree_util.tree_map(
                lambda x: jax.device_get(x[0]), xp_state
            ),
            config=config,
            step=step,
            rng=rng,
        )

        self._save_checkpoint(checkpoint_data, step)

    def _create_unique_hash(self) -> str:
        random_num = datetime.datetime.now().timestamp() + random.random()
        unique_hash = hashlib.md5(str(random_num).encode("utf-8")).hexdigest()
        return unique_hash


class LocalCheckpointer(Checkpointer):
    """A local checkpoint saving and loading class."""

    def load_checkpoint(
        self, checkpoint_path: Optional[str] = None
    ) -> Optional[CkptData]:
        """Loads a checkpoint if any is found."""

        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path

        # Step 1: Load file
        try:
            with open(checkpoint_path, "rb") as checkpoint_file:
                checkpoint_data = pickle.load(checkpoint_file)
                logging.info(
                    "Loading checkpoint from %s, saved at step %d",
                    checkpoint_path,
                    checkpoint_data["step"],
                )

        except FileNotFoundError:
            logging.info("No existing checkpoint found at %s", checkpoint_path)
            return None

        # Retrieve experiment states (params, states etc.)
        experiment_state = checkpoint_data["experiment_state"]
        experiment_state = jax.tree_util.tree_map(
            utils.bcast_local_devices, experiment_state
        )
        return CkptData(
            experiment_state=experiment_state,
            config=checkpoint_data["config"],
            step=checkpoint_data["step"],
            rng=checkpoint_data["rng"],
        )

    def _save_checkpoint(self, checkpoint_data, step):
        """Save checkpoint based on subclass."""

        tmp_path = f"{self._checkpoint_path}_tmp{self._suffix}"
        old_path = f"{self._checkpoint_path}_old{self._suffix}"

        # Creates a rolling ckpt.
        with open(tmp_path, "wb") as checkpoint_file:
            pickle.dump(checkpoint_data, checkpoint_file, protocol=2)

        try:
            os.rename(self.checkpoint_path, old_path)
            remove_old = True
        except FileNotFoundError:
            remove_old = False  # No previous checkpoint to remove

        if remove_old:
            os.remove(old_path)

        os.rename(tmp_path, self.checkpoint_path)
        logging.info("Checkpoint saved at: %s", self.checkpoint_path)
        self._last_checkpoint_step = step

def checkpointer_factory(
    ckpt_type: str, kwargs: types.Config
) -> Union[LocalCheckpointer]:
    """Retrieve a checkpointer given a tag and kwargs."""

    if ckpt_type == types.CheckpointerType.LOCAL:
        ckpt = LocalCheckpointer(**kwargs)
    else:
        raise ValueError(f"Incorrect listener type {ckpt_type}.")

    return ckpt
