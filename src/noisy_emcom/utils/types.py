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

"""Constructs typing of different Objects needed in CIDRE."""

from collections.abc import Iterable
import enum
from typing import Any, Dict, List, NamedTuple, Optional, Union

import chex
import haiku as hk
from ml_collections import config_dict
import optax

Config = Union[Dict[str, Any], config_dict.ConfigDict]
RNNState = chex.ArrayTree
AllParams = Dict[str, List[Optional[hk.Params]]]
AllStates = Dict[str, List[Optional[hk.State]]]
AllOptStates = Dict[str, List[Optional[optax.OptState]]]


class ModularDatasetType:
    STATIC = "static"
    DISJOINT_PARTITIONS = "disjoint_partitions"
    ITERATIVE_PARTITIONS = "iterative_partitions"


class TrainingMode(enum.Enum):
    TRAINING = "training"
    EVAL = "eval"
    EVAL_LG = "evallg"  # RL eval when nr train candidates != nr eval candidates
    EVAL_ILG = "evalilg"
    FORCING = "forcing"


class ImitationMode(enum.Enum):
    BEST = "best"
    RANDOM = "random"
    WORST = "worst"


class ResetMode:
    PAIR = "pair"
    SPEAKER = "speaker"
    LISTENER = "listener"


class Task:
    ATTRIBUTE = "attribute"
    CLASSIFICATION = "classification"
    DISCRIMINATION = "discrimination"
    DISCRIMINATION_DIST = "discriminationdist"
    IMAGES = "images"
    LANDMARK = "landmark"
    MULTICLASSIFICATION = "multiclassification"
    RECONSTRUCTION = "reconstruction"
    REGRESSION = "regression"


class CheckpointerType:
    LOCAL = "local"


class MeaningSimilarity:
    INPUTS = "inputs"
    ATTRIBUTES = "attributes"


class AgentType:
    LEWIS_GAME = "lewis_game"


class RewardType:
    SUCCESS_RATE = "success_rate"
    LOG_PROB = "log_prob"


class ScalerType:
    LINEAR = "linear"


class CoreType:
    ATTENTION = "attention"
    IDENTITY = "identity"
    GRU = "gru"
    LSTM = "lstm"
    RNN_ATTENTION = "rnn_attention"
    STATIC_RNN = "static_rnn"


class TorsoType:
    ATTENTION = "attention"
    DISCRETE = "discrete"
    IDENTITY = "identity"
    MLP = "mlp"
    RECURRENT = "recurrent"


class ListenerHeadType:
    CPC = "cpc"
    DISCRIMINATION = "discrimination"
    DISCRIMINATION_DIST = "discrimination_dist"
    DIST_POLICY_VALUE = "dist_policy_value"
    MERGE_POLICY_VALUE = "merge_policy_value"
    MLP = "mlp"
    MULTIMLP = "multi_mlp"
    NORM_MERGE_POL_VAL = "norm_merge_pol_val"
    RECONSTRUCTION = "recosntruction"
    RECURSIVE_POLICY_VALUE = "recursive_policy_value"


class SpeakerHeadType:
    POLICY = "policy"
    POLICY_VALUE = "policy_value"
    POLICY_QVALUE = "policy_q_value"
    POLICY_QVALUE_DUELING = "policy_q_value_dueling"


class ListenerType:
    SS = "ss"
    RL = "rl"


class ListenerLossType:
    CLASSIF = "classif"
    CPC = "cpc"
    REINFORCE = "reinforce"
    ADV_REINFORCE = "adv_reinforce"
    RECONSTRUCTION = "reconstruction"


class SpeakerLossType:
    REINFORCE = "reinforce"
    POLICYGRADIENT = "policy_gradient"


class ChannelType:
    DEFAULT = "default"
    DETECTABLE = "detectable"
    DETECTABLE_FIXED_TOKENS = "detectable_fixed_tokens"


class ChannelManipulatorLossType:
    REINFORCE = "reinforce"


class ChannelManipulatorType:
    RECURRENT = "recurrent"


class GameRewardType:
    DEFAULT = "default"


class PolicyActuatorType:
    DEFAULT = "default"
    EPSILON_GREEDY = "epsilon_greedy"


class ListenerSsHeadOutputs(NamedTuple):
    predictions: chex.ArrayTree
    targets: Optional[chex.ArrayTree] = None


class ClassifHeadOutputs(NamedTuple):
    predictions: chex.ArrayTree
    targets: Optional[chex.ArrayTree] = None


class RlHeadOutputs(NamedTuple):
    policy_logits: Union[chex.Array, list[chex.Array]]
    q_values: Optional[chex.Array] = None
    value: Optional[chex.Array] = None


class SpeakerHeadOutputs(NamedTuple):
    policy_logits: chex.Array
    q_values: Optional[chex.Array] = None
    value: Optional[chex.Array] = None


class DuelingHeadOutputs(NamedTuple):
    q_values: chex.Array
    value: chex.Array


class Params(NamedTuple):
    speaker: hk.Params
    listener: hk.Params
    target_speaker: Optional[hk.Params]


class States(NamedTuple):
    speaker: hk.State
    listener: hk.State
    target_speaker: Optional[hk.State]


class OptStates(NamedTuple):
    speaker: Optional[optax.OptState] = None
    listener: Optional[optax.OptState] = None


class AgentProperties(NamedTuple):
    params: hk.Params
    opt_states: optax.OptState
    states: hk.State
    target_params: Optional[hk.Params] = None
    target_states: Optional[hk.State] = None


class AllProperties(NamedTuple):
    params: AllParams
    states: AllStates
    opt_states: AllOptStates


class ChannelOutputs(NamedTuple):
    message: chex.Array


class SpeakerOutputs(NamedTuple):
    action: chex.Array
    action_log_prob: chex.Array
    entropy: chex.Array
    policy_logits: chex.Array
    q_values: Optional[chex.Array] = None
    value: Optional[chex.Array] = None


class ListenerSsOutputs(NamedTuple):
    predictions: chex.ArrayTree
    targets: Optional[chex.ArrayTree] = None


class RlOutputs(NamedTuple):
    action: chex.Array
    action_log_prob: chex.Array
    entropy: chex.Array
    policy_logits: chex.Array
    probs: Optional[chex.Array] = None
    q_values: Optional[chex.Array] = None
    value: Optional[chex.Array] = None


class ListenerRlOutputs(NamedTuple):
    action: chex.Array
    action_log_prob: chex.Array
    policy_logits: chex.Array
    entropy: Optional[chex.Array] = None
    probs: Optional[chex.Array] = None
    q_values: Optional[chex.Array] = None
    value: Optional[chex.Array] = None


class ClassifOutputs(NamedTuple):
    predictions: chex.ArrayTree
    targets: Optional[chex.ArrayTree] = None


class EaseOfLearningOutputs(NamedTuple):
    predictions: chex.ArrayTree
    targets: Optional[chex.ArrayTree] = None


class AgentLossOutputs(NamedTuple):
    loss: chex.Array
    stats: Config
    probs: Optional[chex.Array] = None
    reward: Optional[chex.Array] = None


class ListenerLossOutputs(NamedTuple):
    loss: chex.Array
    probs: chex.Array
    stats: Config
    accuracy: Optional[chex.Array] = None
    reward: Optional[chex.Array] = None


class SpeakerLossOutputs(NamedTuple):
    loss: chex.Array
    stats: Config


class DatasetInputs(NamedTuple):
    speaker_inp: chex.Array
    labels: Optional[chex.ArrayTree] = None
    misc: Dict[str, Any] = dict()  # to store debug information


class GamesData(NamedTuple):
    speaker_inp: chex.Array
    round: int = 0
    current_step: Optional[int] = None
    labels: Optional[chex.ArrayTree] = None
    message: Optional[chex.Array] = None
    speaker_outputs: Optional[RlOutputs] = None
    target_speaker_outputs: Optional[RlOutputs] = None
    listener_outputs: Optional[Union[RlOutputs, ClassifOutputs]] = None


class TopSimData(NamedTuple):
    message: chex.Array
    input_meaning: Optional[chex.Array] = None
    label_meaning: Optional[chex.ArrayTree] = None
