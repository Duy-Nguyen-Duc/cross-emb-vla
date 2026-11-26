# Appends action.qpos (robot0_joint_pos + robot0_gripper_qpos, zero-padded to --width) -> demo_data/robomimic/processed.
#   python data/process/robomimic/append_qpos_rs.py [--h5 demo_data/robomimic/source/h5_qpos] [--robots rs_panda rs_iiwa rs_ur5e rs_kinova3]
import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from data.process.label_primitives import fix_stats  # noqa: E402
from data.process.robomimic.pipeline import TASKS  # noqa: E402

H5_NAME = "trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5"
TASK_OF = {v: k for k, v in TASKS.items()}          # instruction -> h5 task directory


def build_robot(src: Path, dst: Path, h5_root: Path, robot: str, width: int, dof: int, tol: float) -> dict:
    s, d = src / robot, dst / robot
    (d / "meta").mkdir(parents=True, exist_ok=True)
    for f in (s / "meta").glob("*"):
        if f.is_file():
            shutil.copy2(f, d / "meta" / f.name)
    if not (d / "videos").exists():
        shutil.copytree((s / "videos").resolve(), d / "videos")   # final dataset: real videos, no links into build/

    eps = [json.loads(l) for l in open(s / "meta/episodes.jsonl")]
    rank, files = Counter(), {}
    stats = {"episodes": 0, "max_state_err": 0.0, "qpos_min": None, "qpos_max": None}
    for e in eps:
        task = TASK_OF[e["tasks"][0]]
        r = rank[task]
        rank[task] += 1
        if task not in files:
            files[task] = h5py.File(h5_root / robot / task / "motionplanning" / H5_NAME, "r")
        g = files[task][f"traj_{r}"]
        pq = next((s / "data").rglob(f"episode_{e['episode_index']:06d}.parquet"))
        df = pd.read_parquet(pq)
        act = np.stack(df["action"].to_numpy()).astype(np.float32)
        st = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
        T = len(act)
        q = g["obs/agent/qpos"][:].astype(np.float32)
        tcp = g["obs/extra/tcp_pose"][:].astype(np.float64)
        assert q.shape[0] in (T, T + 1) and q.shape[1] == dof, (robot, task, r, q.shape, T, dof)
        err = float(np.abs(st[:, :7] - tcp[:T]).max())
        assert err < tol, (robot, task, r, err)
        stats["max_state_err"] = max(stats["max_state_err"], err)
        qp = np.zeros((T, width), np.float32)
        qp[:, :dof] = q[:T]
        lo, hi = q[:T].min(0), q[:T].max(0)
        stats["qpos_min"] = lo if stats["qpos_min"] is None else np.minimum(stats["qpos_min"], lo)
        stats["qpos_max"] = hi if stats["qpos_max"] is None else np.maximum(stats["qpos_max"], hi)
        df["action"] = list(np.concatenate([act, qp], axis=1))
        out = d / "data" / pq.parent.name / pq.name
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        stats["episodes"] += 1
    for f in files.values():
        f.close()

    mp = d / "meta" / "modality.json"
    mod = json.load(open(mp))
    w0 = max(v["end"] for v in mod["action"].values())
    mod["action"]["qpos"] = {"start": w0, "end": w0 + width, "absolute": True}
    json.dump(mod, open(mp, "w"), indent=4)
    fix_stats(d, robot)
    stats["qpos_min"], stats["qpos_max"] = stats["qpos_min"].round(3).tolist(), stats["qpos_max"].round(3).tolist()
    print(f"{robot}: {stats['episodes']} episodes, action width {w0} -> {w0 + width}, dof {dof}, "
          f"max |state - tcp| {stats['max_state_err']:.1e}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="demo_data/robomimic/build/prim")
    ap.add_argument("--dst", default="demo_data/robomimic/processed")
    ap.add_argument("--h5", default="demo_data/robomimic/source/h5_qpos", help="root with <robot>/<task>/motionplanning/" + H5_NAME)
    ap.add_argument("--layout", default="assets/kinematic_graphs/qpos_layout.json")
    ap.add_argument("--robots", nargs="+", default=["rs_panda", "rs_iiwa", "rs_ur5e", "rs_kinova3"])
    ap.add_argument("--width", type=int, default=14, help="padded qpos columns (>= every robot's dof: 9 / 13 / 12 / 13)")
    ap.add_argument("--tol", type=float, default=1e-4, help="max |observation.state[:7] - tcp_pose| accepted")
    args = ap.parse_args()
    layout = json.load(open(args.layout))
    src, dst = Path(args.src), Path(args.dst)
    all_stats = {}
    for r in args.robots:
        dof = len(layout[r]["joints"])
        assert dof <= args.width, f"{r}: {dof} dof > width {args.width}"
        all_stats[r] = build_robot(src, dst, Path(args.h5), r, args.width, dof, args.tol)
        lim = list(zip(layout[r]["joints"], layout[r]["lower"], layout[r]["upper"], all_stats[r]["qpos_min"], all_stats[r]["qpos_max"]))
        out = [f"{n} [{lo:.2f},{hi:.2f}] data [{a:.2f},{b:.2f}]" for n, lo, hi, a, b in lim if a < lo - 0.05 or b > hi + 0.05]
        print(f"{r}: columns outside their limits: {out if out else 'none'}")
    json.dump({"width": args.width, "layout": args.layout, "h5": str(args.h5), "robots": all_stats},
              open(dst / "build_stats.json", "w"), indent=2)
    for f in ("segments.json", "PRIMITIVES_REPORT.json"):            # keep the primitive record with the final data
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)


if __name__ == "__main__":
    main()
