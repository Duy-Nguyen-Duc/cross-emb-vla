# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from dataclasses import dataclass

from data.dataset import ModalityConfig
from data.transform.base import ComposedModalityTransform, ModalityTransform
from data.transform.concat import ConcatTransform
from data.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from model.transforms import GR00TTransform


@dataclass
class BaseDataConfig(ABC):
    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    @abstractmethod
    def transform(self) -> ModalityTransform:
        pass


class PANDAWithRELDataConfig(BaseDataConfig):
    video_keys = ["video.image","video.wrist_image"]
    state_keys = ["state.position","state.rotation","state.gripper"]
    action_keys = ["action.position","action.rotation","action.gripper","action.rel_state_position","action.rel_state_rotation","action.rel_state_gripper","action.progress"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))
    # action keys loaded but not normalized
    no_norm_action_keys = ("action.progress",)

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys
                                     if key not in self.no_norm_action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=10,
                max_action_dim=10,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


class PANDAWithRELPrimDataConfig(PANDAWithRELDataConfig):
    """*_rel + per-frame primitive labels in batch["prim_id"] / batch["prim_prog"]."""
    action_keys = PANDAWithRELDataConfig.action_keys + ["action.prim_id", "action.prim_prog"]
    no_norm_action_keys = ("action.progress", "action.prim_id", "action.prim_prog")


class PANDAWithRELPrimGeoDataConfig(PANDAWithRELPrimDataConfig):
    """*_rel_prim + the raw tcp pose in batch["tcp_pose"]."""

    def transform(self):
        from data.transform.tcp_pose import TcpPosePassthrough
        transforms = list(super().transform().transforms)
        first_state = next(i for i, t in enumerate(transforms)
                           if isinstance(t, StateActionToTensor) and t.apply_to == self.state_keys)
        transforms.insert(first_state, TcpPosePassthrough(apply_to=["state.position", "state.rotation"]))
        return ComposedModalityTransform(transforms=transforms)


class PANDAWithRELPrimQDataConfig(PANDAWithRELPrimDataConfig):
    """*_rel_prim + raw joint positions in batch["qpos"] [T, Q]."""
    action_keys = PANDAWithRELPrimDataConfig.action_keys + ["action.qpos"]
    no_norm_action_keys = PANDAWithRELPrimDataConfig.no_norm_action_keys + ("action.qpos",)


DATA_CONFIG_MAP = {
    # per-tag defaults (experiment/inference_wrapper.py requires the embodiment tag itself as a key)
    'panda_rel': PANDAWithRELDataConfig(),
    'xarm6_rel': PANDAWithRELDataConfig(),
    'widowxai_rel': PANDAWithRELDataConfig(),
    'rs_panda_rel': PANDAWithRELDataConfig(),
    'rs_iiwa_rel': PANDAWithRELDataConfig(),
    'rs_ur5e_rel': PANDAWithRELDataConfig(),
    'rs_kinova3_rel': PANDAWithRELDataConfig(),
    'panda_rel_prim_geo': PANDAWithRELPrimGeoDataConfig(),
    'xarm6_rel_prim_geo': PANDAWithRELPrimGeoDataConfig(),
    'widowxai_rel_prim_geo': PANDAWithRELPrimGeoDataConfig(),
    'panda_rel_prim_q': PANDAWithRELPrimQDataConfig(),
    'xarm6_rel_prim_q': PANDAWithRELPrimQDataConfig(),
    'widowxai_rel_prim_q': PANDAWithRELPrimQDataConfig(),
    'rs_panda_rel_prim_q': PANDAWithRELPrimQDataConfig(),
    'rs_panda_rel_prim_geo': PANDAWithRELPrimGeoDataConfig(),
    'rs_iiwa_rel_prim_q': PANDAWithRELPrimQDataConfig(),
    'rs_iiwa_rel_prim_geo': PANDAWithRELPrimGeoDataConfig(),
    'rs_ur5e_rel_prim_q': PANDAWithRELPrimQDataConfig(),
    'rs_ur5e_rel_prim_geo': PANDAWithRELPrimGeoDataConfig(),
    'rs_kinova3_rel_prim_q': PANDAWithRELPrimQDataConfig(),
    'rs_kinova3_rel_prim_geo': PANDAWithRELPrimGeoDataConfig(),
}
