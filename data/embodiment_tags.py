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

from enum import Enum

# for inference service to map robot_id to embodiment tags
EMBODIMENT_TAGS = {
    # ManiSkill
    "panda_wristcam_rel": "panda_rel",
    "xarm6_robotiq_wristcam_rel": "xarm6_rel",
    "widowxai_wristcam_rel": "widowxai_rel",

    # robosuite
    "rs_panda_rel": "rs_panda_rel",
    "rs_iiwa_rel": "rs_iiwa_rel",
    "rs_ur5e_rel": "rs_ur5e_rel",
    "rs_kinova3_rel": "rs_kinova3_rel",
}

class EmbodimentTag(Enum):

    XARM6_REL = "xarm6_rel"
    PANDA_REL = "panda_rel"
    WIDOWXAI_REL = "widowxai_rel"
    RS_PANDA_REL = "rs_panda_rel"
    RS_IIWA_REL = "rs_iiwa_rel"
    RS_UR5E_REL = "rs_ur5e_rel"
    RS_KINOVA3_REL = "rs_kinova3_rel"

# Embodiment tag string: to projector index in the Action Expert Module (ids are checkpoint slots; keep them fixed)
EMBODIMENT_TAG_MAPPING = {
    EmbodimentTag.XARM6_REL.value: 0,
    EmbodimentTag.PANDA_REL.value: 1,
    EmbodimentTag.WIDOWXAI_REL.value: 3,

    EmbodimentTag.RS_PANDA_REL.value: 10,
    EmbodimentTag.RS_IIWA_REL.value: 11,
    EmbodimentTag.RS_UR5E_REL.value: 12,
    EmbodimentTag.RS_KINOVA3_REL.value: 13,
}
