# ManiSkill data-processing scripts applied to the robosuite demos:
#   python data/process/robomimic/pipeline.py to_lerobot --robot rs_panda [--demo_root demo_data/robomimic/source/h5 --out_root demo_data/robomimic/lerobot --n 200]
#   python data/process/trajectory_canonicalization.py --dataset_path demo_data/robomimic/lerobot --save_path demo_data/robomimic/build/canonical --in_rotation_type quat --out_action_type rel --out_rotation_type quat
#   python data/process/robomimic/pipeline.py segment|label [--robots rs_panda ...]
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

ROBOTS = ["rs_panda", "rs_iiwa", "rs_ur5e", "rs_kinova3"]
TASKS = {"Lift": "Lift the cube.",
         "PickPlaceCan": "Pick up the can and place it in the bin.",
         "NutAssemblySquare": "Pick up the square nut and put it on the square peg."}
OBJECT = {"Lift": "cube", "PickPlaceCan": "can", "NutAssemblySquare": "nut"}
TASK_OF = {"Lift the cube": "Lift", "Pick up the can": "PickPlaceCan", "Pick up the square nut": "NutAssemblySquare"}


def gripper_range(robot_dir: Path):
    """Min / max of obs/extra/gripper_width over every trajectory of the robot (all tasks)."""
    import h5py
    import numpy as np
    lo, hi = np.inf, -np.inf
    for h5 in sorted(robot_dir.glob("*/motionplanning/*.h5")):
        with h5py.File(h5, "r") as f:
            for k in f:
                g = np.asarray(f[k]["obs/extra/gripper_width"])
                lo, hi = min(lo, float(g.min())), max(hi, float(g.max()))
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["to_lerobot", "segment", "label"])
    ap.add_argument("--robot", default=None)
    ap.add_argument("--robots", nargs="+", default=ROBOTS)
    ap.add_argument("--demo_root", default="demo_data/robomimic/source/h5")
    ap.add_argument("--out_root", default="demo_data/robomimic/lerobot")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--h5", default="demo_data/robomimic/source/h5")
    ap.add_argument("--out", default="demo_data/robomimic/build/prim")
    ap.add_argument("--src", default="demo_data/robomimic/build/canonical")
    ap.add_argument("--dst", default="demo_data/robomimic/build/prim")
    args = ap.parse_args()
    if args.cmd == "to_lerobot":
        import data.process.maniskill_to_lerobot as m
        m.TASK_DESCRIPTIONS.update(TASKS)
        lo, hi = gripper_range(Path(args.demo_root) / args.robot)   # fixed range: the evaluator applies the same mapping
        sys.argv = ["maniskill_to_lerobot.py", "--demo_root", args.demo_root, "--robot", args.robot, "--out_root", args.out_root,
                    "--tasks", *TASKS, "--max_episodes_per_task", str(args.n), "--gripper_norm", "fixed",
                    f"--gripper_lo={lo}", f"--gripper_hi={hi}"]
        m.main()
        meta = Path(args.out_root) / args.robot / "meta" / "gripper_range.json"
        meta.write_text(json.dumps({"lo": lo, "hi": hi, "source": "obs/extra/gripper_width over all demos"}) + "\n")
        print(f"{args.robot}: gripper opening range [{lo:.5f}, {hi:.5f}] -> {meta}")
    elif args.cmd == "segment":
        import data.process.segment_primitives as s
        s.ROBOTS, s.TASKS, s.OBJ = list(args.robots), list(TASKS), dict(OBJECT)
        s.GRIP_OPEN = {r: -1.0 for r in args.robots}         # robosuite gripper command: -1 open, +1 close
        s.SPLIT_AT_GRIPPER = True                              # human teleop: the arm keeps moving while the gripper closes
        sys.argv = ["segment_primitives.py", "--h5", args.h5, "--out", args.out]
        s.main()
    else:
        import data.process.label_primitives as l
        l.TASK_OF.update(TASK_OF)
        sys.argv = ["label_primitives.py", "--src", args.src, "--dst", args.dst, "--segments", f"{args.dst}/segments.json",
                    "--robots", *args.robots]
        l.main()


if __name__ == "__main__":
    main()
