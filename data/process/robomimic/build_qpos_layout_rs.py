# qpos_layout.json entries of the robosuite arms (robot0_joint_pos then robot0_gripper_qpos); env robosuite.
#   PYTHONPATH=. python data/process/robomimic/build_qpos_layout_rs.py [--layout assets/kinematic_graphs/qpos_layout.json]
import argparse
import json
import math
import os
import sys

import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from data.process.robomimic.build_kinematic_graphs import ROBOTS, robot_model  # noqa: E402
from model.kinematic_graph import KinematicGraph  # noqa: E402


def layout_for(key, name, graph_dir):
    m, env = robot_model(name)
    robot = env.robots[0]
    names = list(robot.robot_joints) + list(robot.gripper.joints)      # == robot0_joint_pos then robot0_gripper_qpos
    mimic = set()
    for e in range(m.neq):
        if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT:
            mimic.add(int(m.eq_obj1id[e])); mimic.add(int(m.eq_obj2id[e]))
    g = KinematicGraph.load(os.path.join(graph_dir, f"{key}.json"))
    missing = [n for n in names if n not in g.node_names]
    assert not missing, f"{key}: qpos joints without a graph node: {missing}"
    types, lower, upper, mim = [], [], [], []
    for n in names:
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        assert j >= 0, (key, n)
        t = m.jnt_type[j]
        if t == mujoco.mjtJoint.mjJNT_HINGE:
            jt = "revolute" if m.jnt_limited[j] else "continuous"
        elif t == mujoco.mjtJoint.mjJNT_SLIDE:
            jt = "prismatic"
        else:
            raise ValueError((key, n, int(t)))
        lo, hi = (float(m.jnt_range[j][0]), float(m.jnt_range[j][1])) if m.jnt_limited[j] else (0.0, 0.0)
        if hi <= lo:
            lo, hi = (0.0, 0.1) if jt == "prismatic" else (-math.pi, math.pi)
        types.append(jt); lower.append(lo); upper.append(hi); mim.append(j in mimic)
    env.close()
    return {"joints": names, "type": types, "lower": lower, "upper": upper, "mimic": mim, "urdf": f"robosuite:{name}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_dir", default="assets/kinematic_graphs")
    ap.add_argument("--layout", default="assets/kinematic_graphs/qpos_layout.json")
    ap.add_argument("--robots", nargs="+", default=list(ROBOTS))
    args = ap.parse_args()
    layout = json.load(open(args.layout)) if os.path.exists(args.layout) else {}
    for key in args.robots:
        layout[key] = layout_for(key, ROBOTS[key], args.graph_dir)
        print(f"{key}: {len(layout[key]['joints'])} dof {layout[key]['joints']}")
        print("    lower", [round(x, 3) for x in layout[key]["lower"]])
        print("    upper", [round(x, 3) for x in layout[key]["upper"]])
    tmp = args.layout + ".tmp"
    json.dump(layout, open(tmp, "w"), indent=1)
    os.replace(tmp, args.layout)
    print("wrote", args.layout, "robots:", list(layout))


if __name__ == "__main__":
    main()
