import json
import os

import numpy as np

# gym robot_uids -> policy robot uid (mapped to a *_rel embodiment tag)
ROBOT_UID_TO_POLICY_UID = {
    "panda_wristcam": "panda_wristcam_rel",
    "xarm6_robotiq_wristcam": "xarm6_robotiq_wristcam_rel",
    "widowxai_wristcam": "widowxai_wristcam_rel",
}

STEP_LENGTHS = {
    "PickCube-v1": 500,
    "PullCube-v1": 500,
    "PushCube-v1": 500,
    "StackCube-v1": 800,
    "PlaceSphere-v1": 500,
    "LiftPegUpright-v1": 700,
}

# must match how the processed dataset was built
_SPEC = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "maniskill_spec.json")))
TASK_INSTRUCTIONS = _SPEC["task_instructions"]
GRIPPER_QPOS_COLS = _SPEC["gripper_qpos_cols"]
GRIPPER_RANGE = _SPEC["gripper_range"]
ACTION_GRIPPER_DIMS = _SPEC["action_gripper_dims"]


def robot_key(robot_uids: str) -> str:
    for k in GRIPPER_QPOS_COLS:
        if robot_uids.startswith(k):
            return k
    raise KeyError(f"no dataset spec entry for {robot_uids}")


def gripper_state(robot_uids: str, qpos) -> np.ndarray:
    """Gripper channel as in maniskill_to_lerobot.py: driven joints, min-max normalised to [0, 1]."""
    rk = robot_key(robot_uids)
    lo, hi = GRIPPER_RANGE[rk]
    raw = float(np.asarray(qpos)[GRIPPER_QPOS_COLS[rk]].sum())
    return np.array([np.clip((raw - lo) / (hi - lo), 0.0, 1.0)], dtype=np.float32)
