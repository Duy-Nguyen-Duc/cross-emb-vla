# MimicGen source-arm demos (env robosuite): prepare, generate, render.
#   python data/process/robomimic/mimicgen_run.py prepare|generate|render --task Lift [--robot IIWA]
import argparse
import json
import os
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import data.process.robomimic.mimicgen_ext  # noqa: E402,F401  (registers MG_Lift, lift, pick_place_can)
import mimicgen  # noqa: E402

MG = Path(mimicgen.__file__).parent
SRC_H5 = {"Lift": "lift_ph_low_dim.hdf5", "PickPlaceCan": "can_ph_low_dim.hdf5", "NutAssemblySquare": "square_ph_low_dim.hdf5"}
INTERFACE = {"Lift": "MG_Lift", "PickPlaceCan": "MG_PickPlace", "NutAssemblySquare": "MG_Square"}
CONFIG = {"Lift": "lift", "PickPlaceCan": "pick_place_can", "NutAssemblySquare": "square"}
SHORT = {"Lift": "lift", "PickPlaceCan": "can", "NutAssemblySquare": "square"}
ROBOT_KEY = {"Panda": "rs_panda", "IIWA": "rs_iiwa", "UR5e": "rs_ur5e", "Kinova3": "rs_kinova3"}
ROOT = REPO / "demo_data/robomimic/source"


def run_script(script, argv):
    sys.argv = [str(script)] + argv
    runpy.run_path(str(script), run_name="__main__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "generate", "render"])
    ap.add_argument("--task", required=True, choices=list(SRC_H5))
    ap.add_argument("--robot", default=None, choices=list(ROBOT_KEY))
    ap.add_argument("--n", type=int, default=100, help="generate: number of successful demos; render: demos to render")
    ap.add_argument("--src-demos", type=int, default=200, help="prepare, generate: number of Panda source demos")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()
    src_dir = ROOT / "mimicgen/src"
    src_dir.mkdir(parents=True, exist_ok=True)
    src = src_dir / f"{SHORT[args.task]}_src.hdf5"
    if args.cmd == "prepare":
        run_script(MG / "scripts/prepare_src_dataset.py",
                   ["--dataset", str(ROOT / "robomimic" / SRC_H5[args.task]), "--env_interface", INTERFACE[args.task],
                    "--env_interface_type", "robosuite", "--n", str(args.src_demos), "--output", str(src)])
        return
    assert args.robot, "--robot required"
    gen_dir = ROOT / "mimicgen/gen" / f"{SHORT[args.task]}_{args.robot}"
    if args.cmd == "generate":
        from mimicgen.configs.config import config_factory
        import mimicgen.configs.robosuite  # noqa: F401  (registers square etc.)
        config = config_factory(CONFIG[args.task], config_type="robosuite")   # class defaults incl. the task spec
        with config.values_unlocked():
            config.experiment.name = f"{SHORT[args.task]}_{args.robot}"
            config.experiment.source.dataset_path = str(src)
            config.experiment.source.n = args.src_demos   # the demos annotated by prepare
            config.experiment.generation.path = str(gen_dir.parent)
            config.experiment.generation.guarantee = True
            config.experiment.generation.keep_failed = False
            config.experiment.generation.num_trials = args.n
            config.experiment.task.name = None
            config.experiment.task.robot = args.robot
            config.experiment.task.gripper = None
            config.experiment.task.interface = INTERFACE[args.task]
            config.experiment.task.interface_type = "robosuite"
            config.experiment.max_num_failures = max(500, 5 * args.n)
            config.experiment.render_video = False
            config.experiment.num_demo_to_render = 0
            config.experiment.num_fail_demo_to_render = 0
            config.experiment.log_every_n_attempts = 25
            config.experiment.seed = args.seed
            config.obs.collect_obs = True           # low-dim obs only (no cameras): MimicGen's writer needs them; images are rendered later
            config.obs.camera_names = []
        cfg = json.loads(json.dumps(config))
        gen_dir.parent.mkdir(parents=True, exist_ok=True)
        cfg_path = gen_dir.parent / f"{SHORT[args.task]}_{args.robot}_config.json"
        json.dump(cfg, open(cfg_path, "w"), indent=1)
        run_script(MG / "scripts/generate_dataset.py", ["--config", str(cfg_path), "--auto-remove-exp"])
        return
    demo = gen_dir / "demo.hdf5"
    assert demo.exists(), demo
    out_name = f"{ROBOT_KEY[args.robot]}_{SHORT[args.task]}_{args.size}.hdf5"
    import robomimic
    run_script(Path(robomimic.__file__).parent / "scripts/dataset_states_to_obs.py",
               ["--dataset", str(demo), "--output_name", out_name, "--done_mode", "2", "--camera_names", "agentview", "robot0_eye_in_hand",
                "--camera_height", str(args.size), "--camera_width", str(args.size), "--n", str(args.n), "--compress"])
    final = ROOT / "mimicgen" / out_name
    os.replace(gen_dir / out_name, final)
    print("rendered ->", final)


if __name__ == "__main__":
    main()
