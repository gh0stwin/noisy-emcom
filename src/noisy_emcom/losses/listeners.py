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

"""Listener losses."""

import abc
from typing import Any, Optional

import chex
import distrax
import jax
import jax.numpy as jnp
import optax
import rlax

from noisy_emcom.utils import types
from noisy_emcom.utils import utils as emcom_utils
from noisy_emcom.utils.rewards import reward_factory


class ListenerLoss(abc.ABC):
    """Abstract class implementing the listener loss."""

    def __init__(self, reward_type: types.RewardType) -> None:
        self._reward_type = reward_type

    @abc.abstractmethod
    def compute_ensemble_accuracy(
        self, prediction: chex.Array, games: types.GamesData
    ) -> dict[str, Any]:
        pass

    def compute_listener_loss(
        self, rng: chex.PRNGKey, games: types.GamesData
    ) -> types.ListenerLossOutputs:
        """Computes the Listener loss."""

        # Computes the loss.
        output = self._compute_listener_loss(rng=rng, games=games)

        # Turns the loss into reward.
        if self._reward_type == "log_prob":
            reward = -output.loss  # [B]
        elif self._reward_type == "success_rate":
            reward = output.accuracy  # [B]
        else:
            raise ValueError(
                f"Invalid reward reward type {self._reward_type}"
                f"Should be one of [log_prob, success_rate]"
            )

        # Makes the reward non-differentiable.
        reward = jax.lax.stop_gradient(reward)  # [B]

        # Computes global loss and accuracies.
        global_loss = jnp.sum(output.loss, axis=0)
        global_accuracy = jnp.sum(output.accuracy, axis=0)

        # Adds global metrics to stats.
        stats = {
            "listener_loss": global_loss,
            "reward": global_accuracy,
            "accuracy": global_accuracy,
            **output.stats,
        }

        return types.ListenerLossOutputs(
            loss=global_loss,
            probs=output.probs,
            accuracy=global_accuracy,
            reward=reward,
            stats=stats,
        )

    @abc.abstractmethod
    def _compute_listener_loss(
        self, rng: chex.PRNGKey, games: types.GamesData
    ) -> types.ListenerLossOutputs:
        pass


class ClassificationListenerLoss(ListenerLoss):
    """Class implementing the classification loss."""

    def __init__(
        self, reward_type: types.RewardType, task: types.Task, rank_acc: int = 1
    ) -> None:
        super().__init__(reward_type=reward_type)
        self._task = task
        self._rank_acc = rank_acc

    def compute_ensemble_accuracy(
        self, prediction: chex.ArrayTree, games: types.GamesData
    ) -> types.Config:
        """Compute accuracy given a prediction."""
        assert self._task in games.labels
        # labels = games.labels[self._task]  # {str: [B, F]}
        labels = games.listener_outputs.targets

        # Iterates over the attribute to compute an accuracy per attribute.
        accuracy_per_attr = jax.tree_util.tree_map(
            lambda x, y: x == jnp.argmax(y, axis=-1), prediction, labels
        )  # {str: [B]}

        accuracy = jnp.stack(jax.tree_util.tree_leaves(accuracy_per_attr))  # [|{}|, B]
        accuracy = jnp.mean(accuracy, axis=0)  # [B]

        return dict(ensemble_acc=jnp.sum(accuracy, axis=0))

    def _compute_listener_loss(
        self, rng: chex.PRNGKey, games: types.GamesData
    ) -> types.ListenerLossOutputs:
        """Computes the Listener loss."""
        del rng  # Deterministic loss

        predictions = games.listener_outputs.predictions  # {str: [B, F]}
        assert self._task in games.labels
        targets = games.listener_outputs.targets  # {str: [B, F]}
        #  labels = games.labels[self._task]  # {str: [B, F]}

        # Iterates over the attribute to compute an accuracy per attribute.
        accuracy_per_attr = jax.tree_util.tree_map(
            lambda x, y: jnp.argmax(x, axis=-1) == jnp.argmax(y, axis=-1),
            predictions,
            targets,
        )  # {str: [B]}

        global_accuracy = jnp.stack(
            jax.tree_util.tree_leaves(accuracy_per_attr), axis=0
        )  # [|{}|, B]

        global_accuracy = jnp.mean(global_accuracy, axis=0)  # [B]
        listener_probs = jax.tree_util.tree_map(jax.nn.softmax, predictions)
        listener_loss = jax.tree_util.tree_map(
            emcom_utils.softmax_cross_entropy, predictions, targets
        )  # {str: [B]}

        listener_loss = jnp.stack(
            jax.tree_util.tree_leaves(listener_loss), axis=0
        )  # [|{}|, B]

        listener_loss = jnp.mean(listener_loss, axis=0)  # [B]

        # Sums over the batch size.
        accuracy_per_attr = jax.tree_util.tree_map(jnp.sum, accuracy_per_attr)

        stats = {f"accuracy_{k}": v for k, v in accuracy_per_attr.items()}

        if self._rank_acc > 1:

            def rank_acc(x, y):
                x = jnp.argsort(x)[:, -self._rank_acc :]
                y = jnp.argmax(y, axis=-1, keepdims=True)
                acc = jnp.sum(x == y, axis=-1)
                return acc.astype(bool)

            rank_accuracy_per_attr = jax.tree_util.tree_map(
                rank_acc, predictions, targets
            )

            global_rank_accuracy = jnp.stack(
                jax.tree_util.tree_leaves(rank_accuracy_per_attr), axis=0
            )  # [|{}|, B]

            global_rank_accuracy = jnp.mean(global_rank_accuracy, axis=0)  # [B]
            global_rank_accuracy = jnp.sum(global_rank_accuracy, axis=0)
            stats |= {f"rank_{self._rank_acc}_accuracy": global_rank_accuracy}

        return types.ListenerLossOutputs(
            loss=listener_loss,
            probs=listener_probs,
            accuracy=global_accuracy,
            stats=stats,
        )


class CpcListenerLoss(ListenerLoss):
    """Class implementing the CPC loss."""

    def __init__(
        self, reward_type: types.RewardType, num_distractors: int, cross_device: bool
    ) -> None:
        super().__init__(reward_type=reward_type)
        self._num_distractors = num_distractors
        self._cross_device = cross_device

    def compute_ensemble_accuracy(self, prediction, games):
        """Computes accuracy given a prediction."""
        del games
        effective_batchsize = prediction.shape[0]
        num_distractors = self._num_distractors
        if num_distractors >= (effective_batchsize - 1):
            num_distractors = -1
        if num_distractors == -1:
            accuracy = prediction == jnp.arange(effective_batchsize)
        else:
            accuracy = prediction == 0
        # Transforms accuracy from bool to integer.
        accuracy = accuracy * 1
        return dict(ensemble_acc=jnp.sum(accuracy, axis=0))

    def _compute_listener_loss(
        self, rng: chex.PRNGKey, games: types.GamesData
    ) -> types.ListenerLossOutputs:
        """Computes CPC loss."""
        effective_batchsize, feature_dim = games.listener_outputs.targets.shape

        # Warning: at evaluation time, batch size is small.
        # Use all the batch as distractors at eval time.
        if self._num_distractors >= (effective_batchsize - 1):
            self._num_distractors = -1

        if self._num_distractors == -1:
            # Computes CPC on the full batch.
            predictions = games.listener_outputs.predictions
            targets = games.listener_outputs.targets
            batch_indices = jnp.arange(effective_batchsize)

            # If we are on multiple devices we have to gather targets from other
            # devices and offset batch indices by the device id.
            # We do not pmap the init to gain compilation time so we do not gather
            # across devices at init.
            if jax.device_count() > 1 and self._cross_device:
                targets = jax.lax.all_gather(
                    targets, axis_name="i"
                )  # Num_devices, B, F
                targets = targets.reshape(-1, feature_dim)  # Large_Batch, F
                global_batch_indices = (
                    batch_indices + jax.lax.axis_index("i") * effective_batchsize
                )
            else:
                global_batch_indices = batch_indices

            cosine_sim = -emcom_utils.cosine_loss(
                predictions[:, None, :], targets[None, :, :]
            )

            listener_probs = jax.nn.softmax(cosine_sim, axis=-1)
            listener_loss = -jax.nn.log_softmax(cosine_sim, axis=-1)[
                batch_indices, global_batch_indices
            ]

            accuracy = jnp.argmax(cosine_sim, axis=-1) == global_batch_indices
        else:
            # Computes CPC on a predefined numbner of distractors.
            batch_distractors = []
            for i in range(effective_batchsize):
                key_rng, rng = jax.random.split(rng)
                potential_distractors_idx = list(range(effective_batchsize))
                potential_distractors_idx.remove(i)
                distractor_idx = jax.random.choice(
                    key_rng,
                    jnp.array(potential_distractors_idx),
                    shape=[self._num_distractors],
                    replace=False,
                )
                distractors = jnp.take(
                    games.listener_outputs.targets, distractor_idx, axis=0
                )
                target = games.listener_outputs.targets[i : (i + 1)]
                batch_distractors.append(jnp.concatenate([target, distractors], axis=0))

            targets = jnp.stack(batch_distractors, axis=0)
            cosine_sim = -emcom_utils.cosine_loss(
                games.listener_outputs.predictions[:, None, :], targets
            )
            listener_probs = jax.nn.softmax(cosine_sim, axis=-1)
            # By construction the target is in position 0.
            listener_loss = -jax.nn.log_softmax(cosine_sim, axis=-1)[:, 0]
            accuracy = jnp.argmax(cosine_sim, axis=-1) == 0
        # Transforms accuracy from bool to integer.
        accuracy = accuracy * 1

        return types.ListenerLossOutputs(
            loss=listener_loss,
            probs=listener_probs,
            accuracy=accuracy,
            stats=dict(),
        )


class ReconstructionLoss(ListenerLoss):
    def __init__(self, reward_type: types.RewardType) -> None:
        super().__init__(reward_type)
        self._task = types.Task.RECONSTRUCTION

    def compute_ensemble_accuracy(
        self, prediction: chex.ArrayTree, games: types.GamesData
    ) -> types.Config:
        """Compute accuracy given a prediction."""
        return dict(ensemble_acc=0)

    def _compute_listener_loss(
        self, rng: chex.PRNGKey, games: types.GamesData
    ) -> types.ListenerLossOutputs:
        """Computes the Listener loss."""
        del rng  # Deterministic loss
        assert self._task in games.labels

        predictions = games.listener_outputs.predictions  # [B, H, W, C]
        targets = games.listener_outputs.targets  # [B, H, W, C]
        loss = 0.5 * jnp.square(predictions - targets)
        loss = jnp.sum(loss, axis=(-3, -2, -1))
        return types.ListenerLossOutputs(
            loss=loss, probs=0, accuracy=jnp.zeros(targets.shape[0]), stats={}
        )


class ActorCriticLoss(ListenerLoss):
    """Base class for a AC loss."""

    def __init__(self, critic_weight: float = 1) -> None:
        super().__init__(types.RewardType.SUCCESS_RATE)
        self._critic_weight = critic_weight

    def compute_listener_loss(
        self, rng: chex.PRNGKey, games: types.GamesData
    ) -> types.ListenerLossOutputs:
        return self._compute_listener_loss(rng, games)

    @abc.abstractmethod
    def compute_actor_loss(
        self,
        games: types.GamesData,
        reward: chex.Array,
        sg_values: dict,
        masks: dict[str, chex.Array],
    ) -> tuple[chex.Array, dict]:
        pass

    @abc.abstractmethod
    def compute_critic_loss(
        self, games: types.GamesData, reward: chex.Array
    ) -> tuple[chex.Array, dict, dict]:
        pass

    @abc.abstractmethod
    def compute_reward(
        self, games: types.GamesData, rng: chex.PRNGKey
    ) -> tuple[chex.Array, dict, dict]:
        pass

    def _compute_listener_loss(
        self, rng: chex.PRNGKey, games: types.GamesData
    ) -> types.ListenerLossOutputs:
        assert isinstance(games.listener_outputs, types.ListenerRlOutputs)

        reward, masks, reward_stats = self.compute_reward(games, rng)
        critic_loss, sg_values, critic_stats = self.compute_critic_loss(games, reward)
        actor_loss, actor_stats = self.compute_actor_loss(
            games, reward, sg_values, masks
        )

        listener_loss = self._critic_weight * critic_loss + actor_loss
        stats = dict(listener_loss=jax.lax.stop_gradient(listener_loss))
        policy_logits = games.listener_outputs.policy_logits

        return types.ListenerLossOutputs(
            loss=listener_loss,
            probs=jax.tree_util.tree_map(
                jax.nn.softmax, jax.lax.stop_gradient(policy_logits)
            ),
            reward=reward,
            stats=stats | critic_stats | actor_stats | reward_stats,
        )


class ReinforceLoss(ActorCriticLoss):
    """Class implementing Reinforce loss for listener agent."""

    def __init__(
        self,
        reward_config: types.Config,
        entropy_scheduler: str,
        entropy_scheduler_kwargs: types.Config,
        use_baseline: bool = False,
        critic_weight: float = 1,
    ) -> None:
        super().__init__(critic_weight)
        self.reward = reward_factory(**reward_config)
        self._use_baseline = use_baseline
        self._entropy_coeff = getattr(optax, entropy_scheduler)(
            **entropy_scheduler_kwargs
        )

    def compute_ensemble_accuracy(
        self, prediction: chex.Array, games: types.GamesData
    ) -> dict[str, Any]:
        accuracy = prediction == jnp.arange(prediction.shape[0])
        accuracy *= 1
        return dict(ensemble_acc=jnp.sum(accuracy, axis=0))

    def compute_actor_loss(
        self,
        games: types.GamesData,
        reward: chex.Array,
        sg_values: dict,
        masks: dict[str, chex.Array],
    ) -> tuple[chex.Array, dict]:
        action_log_prob = games.listener_outputs.action_log_prob
        policy_loss = (reward - sg_values["state"]) * action_log_prob
        policy_loss = -1 * jnp.sum(policy_loss, axis=0)
        entropy = games.listener_outputs.entropy
        entropy_sum = jnp.sum(entropy, axis=0)
        entropy_loss = -1 * self._entropy_coeff(games.current_step) * entropy_sum

        return (
            policy_loss + entropy_loss,
            dict(
                listener_policy_loss=policy_loss,
                listener_entropy_loss=entropy_loss,
                listener_entropy=entropy_sum,
                listener_entropy_success=jnp.sum(masks["success"] * entropy),
                listener_entropy_failure=jnp.sum(masks["failure"] * entropy),
            ),
        )

    def compute_critic_loss(
        self, games: types.GamesData, reward: chex.Array, **kwargs
    ) -> tuple[chex.Array, dict, dict]:
        if not self._use_baseline:
            value = sg_value = value_loss = value_stats = 0.0
        else:
            assert games.listener_outputs.value is not None
            # From [B, F] to [B], where dim F = 1
            value = jnp.squeeze(games.listener_outputs.value, axis=-1)
            sg_value = jax.lax.stop_gradient(value)
            value_loss = jnp.sum(jnp.square(reward - value), axis=0)
            value_stats = jnp.sum(sg_value, axis=0)

        return (
            value_loss,
            dict(state=sg_value),
            dict(listener_value=value_stats, listener_value_loss=value_loss),
        )

    def compute_reward(
        self, games: types.GamesData, rng: chex.PRNGKey
    ) -> tuple[chex.Array, dict, dict]:
        action = games.listener_outputs.action
        reward_dtype = games.listener_outputs.policy_logits.dtype
        mask = action == jnp.arange(action.shape[0])
        reward = jnp.full_like(action, self.reward.failure(games), dtype=reward_dtype)
        reward = jnp.where(mask, self.reward.success(games), reward)
        reward_range = self.reward.limits[1] - self.reward.limits[0]
        norm_reward = (reward - self.reward.limits[0]) / reward_range

        return (
            reward,
            dict(success=mask, failure=~mask),
            dict(reward=norm_reward.sum()),
        )


class AdvantageReinforceLoss(ReinforceLoss):
    def compute_actor_loss(
        self,
        games: types.GamesData,
        reward: chex.Array,
        sg_values: dict,
        masks: dict[str, chex.Array],
    ) -> tuple[chex.Array, dict]:
        action_log_prob = games.listener_outputs.action_log_prob
        policy_loss = (sg_values["action"] - sg_values["state"]) * action_log_prob
        policy_loss = -1 * jnp.sum(policy_loss, axis=0)
        entropy = games.listener_outputs.entropy
        entropy_sum = jnp.sum(entropy, axis=0)
        entropy_loss = -1 * self._entropy_coeff(games.current_step) * entropy_sum

        return (
            policy_loss + entropy_loss,
            dict(
                listener_policy_loss=policy_loss,
                listener_entropy_loss=entropy_loss,
                listener_entropy=entropy_sum,
                listener_entropy_success=jnp.sum(masks["success"] * entropy),
                listener_entropy_failure=jnp.sum(masks["failure"] * entropy),
            ),
        )

    def compute_critic_loss(
        self, games: types.GamesData, reward: chex.Array, **kwargs
    ) -> tuple[chex.Array, chex.Array, dict]:
        assert games.listener_outputs.q_values is not None
        q_values = games.listener_outputs.q_values
        action = games.listener_outputs.action
        q_value_chosen = rlax.batched_index(q_values, action)
        sg_q_value_chosen = jax.lax.stop_gradient(q_value_chosen)
        action_value_loss = jnp.sum(jnp.square(reward - q_value_chosen), axis=0)
        value_loss, sg_value, value_stats = 0, 0, 0

        if self._use_baseline:
            assert games.listener_outputs.value is not None
            # From [B, F] to [B], where dim F = 1
            value = jnp.squeeze(games.listener_outputs.value, axis=-1)
            sg_value = jax.lax.stop_gradient(value)
            value_stats = jnp.sum(sg_value, axis=0)
            value_loss = jnp.sum(jnp.square(reward - value), axis=0)

        return (
            action_value_loss + value_loss,
            dict(action=sg_q_value_chosen, state=sg_value),
            dict(
                listener_q_value_chosen_stats=jnp.sum(sg_q_value_chosen, axis=0),
                listener_value=value_stats,
                listener_action_value_loss=action_value_loss,
                listener_value_loss=value_loss,
            ),
        )


class ActorCriticLossWrapper(ActorCriticLoss):
    def __init__(self, loss: ActorCriticLoss, critic_weight: float = 1) -> None:
        super().__init__(critic_weight)
        self._loss = loss

    def compute_ensemble_accuracy(
        self, prediction: chex.Array, games: types.GamesData
    ) -> dict[str, Any]:
        return self._loss.compute_ensemble_accuracy(prediction, games)

    def compute_actor_loss(
        self,
        games: types.GamesData,
        reward: chex.Array,
        sg_values: chex.Array,
        masks: dict[str, chex.Array],
    ) -> tuple[chex.Array, dict]:
        return self._loss.compute_actor_loss(games, reward, sg_values, masks)

    def compute_critic_loss(
        self, games: types.GamesData, reward: chex.Array
    ) -> tuple[chex.Array, dict, dict]:
        return self._loss.compute_critic_loss(games, reward)

    def compute_reward(
        self, games: types.GamesData, rng: chex.PRNGKey
    ) -> tuple[chex.Array, dict, dict]:
        return self._loss.compute_reward(games)


def listener_loss_factory(
    loss_type: types.ListenerLossType,
    kwargs: types.Config,
    wrappers: Optional[list[str]] = None,
    wrapper_kwargs: Optional[list[dict[str, Any]]] = None,
    common_kwargs: Optional[dict] = None,
) -> ListenerLoss:
    """Factory to select the listener's loss."""

    if wrappers is None:
        wrappers = []

    if wrapper_kwargs is None:
        wrapper_kwargs = []

    if common_kwargs is None:
        common_kwargs = {}

    loss_specific_kwargs = kwargs.get(loss_type, dict())
    all_kwargs = {**common_kwargs, **loss_specific_kwargs}

    if loss_type == types.ListenerLossType.CLASSIF:
        listener_loss = ClassificationListenerLoss(**all_kwargs)
    elif loss_type == types.ListenerLossType.CPC:
        listener_loss = CpcListenerLoss(**all_kwargs)
    elif loss_type == types.ListenerLossType.RECONSTRUCTION:
        listener_loss = ReconstructionLoss(**all_kwargs)
    elif loss_type == types.ListenerLossType.REINFORCE:
        listener_loss = ReinforceLoss(**all_kwargs)
    elif loss_type == types.ListenerLossType.ADV_REINFORCE:
        listener_loss = AdvantageReinforceLoss(**all_kwargs)
    else:
        raise ValueError(f"Incorrect listener loss type {loss_type}.")

    return listener_loss
