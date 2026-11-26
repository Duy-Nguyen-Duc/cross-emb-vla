# Rendered robomimic hdf5 -> ManiSkill trajectory layout (<out>/<robot>/<task>/motionplanning/*.h5).
#   python data/process/robomimic/to_maniskill_h5.py --src demo_data/robomimic/source/robomimic/lift_ph_256.hdf5 --robot rs_panda --task Lift [--no-images --out demo_data/robomimic/source/h5_qpos]
import argparse
from pathlib import Path

import h5py
import numpy as np

OBJECT = {"Lift": "cube", "PickPlaceCan": "can", "NutAssemblySquare": "nut"}
GRIPPERS = {"rs_panda": "panda", "rs_iiwa": "robotiq", "rs_ur5e": "robotiq", "rs_kinova3": "robotiq"}


def xyzw_to_wxyz(q):
    q = np.asarray(q, np.float32)
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def gripper_opening(qpos, kind):
    """Monotone opening signal: Panda = finger distance; Robotiq = minus the driven knuckle angle (0 open)."""
    qpos = np.asarray(qpos, np.float32)
    if kind == "panda":
        return (qpos[:, 0] - qpos[:, 1])[:, None]
    return (-qpos[:, 0])[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="robomimic hdf5 with rendered image observations")
    ap.add_argument("--robot", required=True, choices=list(GRIPPERS))
    ap.add_argument("--task", required=True, choices=list(OBJECT))
    ap.add_argument("--out", default="demo_data/robomimic/source/h5")
    ap.add_argument("--no-images", action="store_true", help="skip the camera images (lean file, e.g. --out demo_data/robomimic/source/h5_qpos)")
    args = ap.parse_args()
    dst = Path(args.out) / args.robot / args.task / "motionplanning" / "trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5"
    dst.parent.mkdir(parents=True, exist_ok=True)
    kind, obj = GRIPPERS[args.robot], OBJECT[args.task]
    with h5py.File(args.src, "r") as f, h5py.File(dst, "w") as g:
        demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
        lens = []
        for i, name in enumerate(demos):
            d = f["data"][name]
            o = d["obs"]
            T = d["actions"].shape[0]
            t = g.create_group(f"traj_{i}")
            t.create_dataset("actions", data=np.asarray(d["actions"], np.float32))
            t.create_dataset("obs/extra/tcp_pose", data=np.concatenate(
                [np.asarray(o["robot0_eef_pos"], np.float32), xyzw_to_wxyz(o["robot0_eef_quat"])], axis=-1))
            t.create_dataset("obs/extra/gripper_width", data=gripper_opening(o["robot0_gripper_qpos"], kind))
            t.create_dataset("obs/agent/qpos", data=np.concatenate(
                [np.asarray(o["robot0_joint_pos"], np.float32), np.asarray(o["robot0_gripper_qpos"], np.float32)], axis=-1))
            for src_key, cam in () if args.no_images else (("agentview_image", "third_view_camera"), ("robot0_eye_in_hand_image", "hand_camera")):
                img = np.asarray(o[src_key], np.uint8)
                t.create_dataset(f"obs/sensor_data/{cam}/rgb", data=np.ascontiguousarray(img), compression="gzip", compression_opts=1)
            ob = np.asarray(o["object"], np.float32)
            t.create_dataset(f"env_states/actors/{obj}", data=np.concatenate([ob[:, :3], xyzw_to_wxyz(ob[:, 3:7])], axis=-1))
            t.create_dataset("success", data=np.ones(T, bool))
            lens.append(T)
        g.attrs["source"] = str(Path(args.src).resolve())
    print(f"{args.robot}/{args.task}: {len(lens)} trajectories, steps min/mean/max {min(lens)}/{np.mean(lens):.1f}/{max(lens)} -> {dst}")


if __name__ == "__main__":
    main()
