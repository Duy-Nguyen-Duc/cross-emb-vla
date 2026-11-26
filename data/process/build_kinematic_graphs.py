import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model.kinematic_graph import parse_urdf  # noqa: E402

MS = os.environ.get("MANISKILL_ASSETS",
                    os.path.join(os.environ.get("MANISKILL_ROOT", "../ManiSkill"), "mani_skill/assets/robots"))
MS_DATA = os.environ.get("MANISKILL_DATA",
                         os.path.join(os.environ.get("MS_ASSET_DIR", os.path.expanduser("~/.maniskill")), "data/robots"))
DEFAULT_URDFS = {
    "panda": f"{MS}/panda/panda_v2.urdf",
    "xarm6": f"{MS}/xarm6/xarm6_robotiq.urdf",
    "widowxai": f"{MS_DATA}/widowxai/wxai_base.urdf",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/kinematic_graphs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for robot, path in DEFAULT_URDFS.items():
        g = parse_urdf(path, robot=robot, bidirectional=True)
        g.save(os.path.join(args.out, f"{robot}.json"))
        movable = sum(1 for f in g.node_feats if f[3] == 0.0)
        grip = sum(1 for f in g.node_feats if f[-3] == 1.0)
        print(f"{robot:9s} nodes {g.num_nodes:3d} (movable {movable:2d}, gripper-named {grip:2d})  "
              f"edges {len(g.edge_src):3d}  <- {path}")
        print("          ", " ".join(g.node_names))


if __name__ == "__main__":
    main()
