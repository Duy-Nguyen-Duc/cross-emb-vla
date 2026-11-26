# Kinematic graphs of the robosuite arms from the compiled MuJoCo model (env robosuite).
#   PYTHONPATH=. python data/process/robomimic/build_kinematic_graphs.py [--out assets/kinematic_graphs]
import argparse
import math
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from model.kinematic_graph import (EDGE_FEATURE_NAMES, JOINT_TYPES, NODE_FEATURE_NAMES, GRIPPER_WORDS,  # noqa: E402
                                   KinematicGraph, _quat_conj, _quat_rotate)

ROBOTS = {"rs_panda": "Panda", "rs_iiwa": "IIWA", "rs_ur5e": "UR5e", "rs_kinova3": "Kinova3"}


def robot_model(name):
    import robosuite
    from robosuite.controllers import load_controller_config
    env = robosuite.make("Lift", robots=name, controller_configs=load_controller_config(default_controller="OSC_POSE"),
                         has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
    env.reset()
    return env.sim.model._model, env


def build(robot_key, robot_name):
    m, env = robot_model(robot_name)
    bname = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
    jname = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    bodies = [i for i in range(m.nbody) if (bname(i) or "").startswith(("robot0_", "gripper0_"))]
    root = next(i for i in bodies if bname(i) == "robot0_base")
    children = {b: [c for c in bodies if m.body_parentid[c] == b] for b in bodies}
    mimic = set()
    for e in range(m.neq):
        if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT:
            mimic.add(int(m.eq_obj1id[e])); mimic.add(int(m.eq_obj2id[e]))

    node_names, node_info, parent_node = [bname(root)], [dict(type="fixed")], [None]
    edges = []   # (parent_node, child_node, xyz, quat)

    def add_node(name, info, parent, xyz, quat):
        node_names.append(name); node_info.append(info); parent_node.append(parent)
        edges.append((parent, len(node_names) - 1, list(map(float, xyz)), list(map(float, quat))))
        return len(node_names) - 1

    def visit(body, parent):
        last = parent
        xyz, quat = m.body_pos[body], m.body_quat[body]
        nj = int(m.body_jntnum[body])
        if nj == 0:
            last = add_node(bname(body), dict(type="fixed"), last, xyz, quat)
        for k in range(nj):
            j = int(m.body_jntadr[body]) + k
            t = m.jnt_type[j]
            if t == mujoco.mjtJoint.mjJNT_HINGE:
                jt = "revolute" if m.jnt_limited[j] else "continuous"
            elif t == mujoco.mjtJoint.mjJNT_SLIDE:
                jt = "prismatic"
            else:
                jt = "fixed"
            lo, up = (float(m.jnt_range[j][0]), float(m.jnt_range[j][1])) if m.jnt_limited[j] else (0.0, 0.0)
            info = dict(type=jt, lower=lo, upper=up, axis=[float(a) for a in m.jnt_axis[j]], mimic=j in mimic)
            last = add_node(jname(j), info, last, xyz if k == 0 else [0, 0, 0], quat if k == 0 else [1, 0, 0, 0])
        for c in children[body]:
            visit(c, last)

    for c in children[root]:
        visit(c, 0)
    env.close()

    n = len(node_names)
    depth = [0] * n
    for i in range(1, n):
        d, p = 0, parent_node[i]
        while p is not None and p != 0:
            d += 1; p = parent_node[p]
        depth[i] = d + 1
    max_depth = max(1, max(depth))
    has_child = {parent_node[i] for i in range(1, n)}
    node_feats = []
    for i, name in enumerate(node_names):
        info = node_info[i]
        jt = info["type"]
        lo, up = info.get("lower", 0.0), info.get("upper", 0.0)
        if jt == "continuous":
            lo, up = -math.pi, math.pi
        prismatic = jt == "prismatic"
        scale = 0.1 if prismatic else math.pi
        feats = [1.0 if jt == t else 0.0 for t in JOINT_TYPES]
        feats += [lo / scale, up / scale, (up - lo) / scale, 0.0, 0.0]          # no velocity / effort limits in MJCF
        feats += info.get("axis", [0.0] * 3) if jt != "fixed" else [0.0] * 3
        grip = any(w in name.lower() for w in GRIPPER_WORDS) or name.startswith("gripper0_")
        feats += [depth[i] / max_depth, float(info.get("mimic", False)), float(grip and i > 0),
                  float(i == 0), float(i not in has_child)]
        node_feats.append([float(x) for x in feats])
    edge_src, edge_dst, edge_feats = [], [], []
    for p, c, xyz, q in edges:
        length = float(np.linalg.norm(xyz))
        edge_src.append(p); edge_dst.append(c); edge_feats.append([*xyz, *q, length, 0.0])
        qi = _quat_conj(q)
        t_inv = [-x for x in _quat_rotate(qi, xyz)]
        edge_src.append(c); edge_dst.append(p); edge_feats.append([*t_inv, *qi, length, 1.0])
    return KinematicGraph(robot=robot_key, urdf=f"robosuite:{robot_name}", node_names=node_names, node_feats=node_feats,
                          edge_src=edge_src, edge_dst=edge_dst, edge_feats=[[float(x) for x in e] for e in edge_feats],
                          node_feature_names=list(NODE_FEATURE_NAMES), edge_feature_names=list(EDGE_FEATURE_NAMES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/kinematic_graphs")
    ap.add_argument("--robots", nargs="+", default=list(ROBOTS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for key in args.robots:
        g = build(key, ROBOTS[key])
        g.save(os.path.join(args.out, f"{key}.json"))
        movable = sum(1 for f in g.node_feats if f[3] == 0.0)
        grip = sum(1 for f in g.node_feats if f[-3] == 1.0)
        print(f"{key:10s} nodes {g.num_nodes:3d} (movable {movable:2d}, gripper {grip:2d}) edges {len(g.edge_src):3d}")
        print("           ", " ".join(g.node_names))


if __name__ == "__main__":
    main()
