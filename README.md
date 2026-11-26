# Motion-Aware Kinematic Graphs for Cross-Embodiment Motif Learning

This repository contains the full pipeline of the motion-aware kinematic graph method: demonstration generation, data processing, the three training stages, and the cross-embodiment benchmarks on ManiSkill 3 and robomimic, with and without perception steering.

The method has three stages:

| Stage | Model | What it learns |
| --- | --- | --- |
| 1 | Graph-routed motif tokenizer (`model/stage1_graph.py`) | Discrete action motifs. Each robot is a kinematic graph whose nodes and edges also encode the joint trajectories inside the window; the graph routes every motif token to the codebook. |
| 2 | Motif predictor (`model/stage2_3.py`) | Predicts the motif tokens from images and language, on top of the frozen stage-1 tokenizer. |
| 3 | E-LoRA policy (`model/stage3_elora.py`) | Flow-matching action head conditioned on the predicted motifs, with embodiment-specific LoRA updates selected through the kinematic graph, and geometry tokens from the wrist view (`model/geometry_tokens.py`). |

At test time the policy can be combined with training-free perception steering.

## Table of Contents

*   🧩 [Benchmarks](#benchmarks)
*   🗂️ [Repository layout](#repository-layout)
*   💻 [Requirements](#requirements)
*   🛠️ [Environment setup](#environment-setup)
    *   [Main environment](#main-environment)
    *   [Software Vulkan renderer](#software-vulkan-renderer)
    *   [robosuite environment (robomimic benchmark)](#robosuite-environment-robomimic-benchmark)
*   📦 [Prepare the data](#prepare-the-data)
    *   [ManiSkill](#maniskill)
    *   [robomimic](#robomimic)
    *   [Real-robot data](#real-robot-data)
    *   [Kinematic graphs and camera calibration](#kinematic-graphs-and-camera-calibration)
*   🔥 [Training](#training)
    *   [One command per K](#one-command-per-k)
    *   [Running the stages by hand](#running-the-stages-by-hand)
*   🎮 [Evaluation](#evaluation)
    *   [ManiSkill: Transfer and Global](#maniskill-transfer-and-global)
    *   [robomimic](#robomimic-1)
    *   [Perception steering](#perception-steering)
    *   [Evaluating a downloaded checkpoint](#evaluating-a-downloaded-checkpoint)
*   📁 [Output directories](#output-directories)
*   📝 [Reproducibility notes](#reproducibility-notes)
*   🙏 [Acknowledgements](#acknowledgements)

## Benchmarks

**ManiSkill 3.** Three robots and six tasks, all with a wrist camera and a third-view camera:

| Robot (`robot_uids`) | Transfer tasks (K demos) | Seen tasks (50 demos) |
| --- | --- | --- |
| Franka Panda (`panda_wristcam`) | PushCube, PlaceSphere | PullCube, LiftPegUpright, PickCube, StackCube |
| UFactory xArm6 + Robotiq (`xarm6_robotiq_wristcam`) | PullCube, LiftPegUpright | PushCube, PlaceSphere, PickCube, StackCube |
| WidowX AI (`widowxai_wristcam`) | PickCube, StackCube | PushCube, PlaceSphere, PullCube, LiftPegUpright |

Each robot sees only **K** demonstrations (K = 1, 3, 5, 10, 50) of its two transfer tasks and 50 demonstrations of every other task. K selects the first K episodes of each transfer pair (`modifiy_tasks_num` in `config/dataset/dataset_maniskill_prim_q.yaml`).

*   **Transfer**: the 6 transfer pairs x 50 episodes = 300 episodes.
*   **Global**: all 18 robot-task pairs x 50 episodes = 900 episodes.

Protocol: environment seeds 2025-2074, policy seed 7 with a SHA-256 seed per request, 4 denoising steps, the full 16-action chunk executed per prediction, `pd_ee_delta_pose` control at 20 Hz, 256x256 cameras.

**robomimic.** Lift, PickPlaceCan and NutAssemblySquare in robosuite. The target robot is the Franka Panda with K = 50 proficient-human demonstrations per task (the first 50 of the 200 robomimic PH demos). The source robots IIWA, UR5e and Kinova3 (Robotiq grippers) each have 100 MimicGen demonstrations per task, generated from the Panda demos. The benchmark runs the Panda on the three tasks, 150 episodes each (450 episodes): episode seeds 2025-2174, policy seed 7, horizon 400 steps, OSC_POSE delta control at 20 Hz, 256x256 agentview and eye-in-hand cameras.

## Repository layout

```plaintext
assets/
  kinematic_graphs/      kinematic graphs of all robots + joint layout (qpos_layout.json)
  wrist_camera/          camera intrinsics and mounts (wrist_cameras.json: ManiSkill, wrist_cameras_rs.json: robosuite)
config/
  train_stage1_graph.yaml, train_stage2.yaml, train_stage3.yaml   Hydra entry configs
  dataset/dataset_maniskill_prim_q.yaml, dataset_robomimic_prim_q.yaml   datasets and K-shot splits
  models/model_graph_elora_geod.yaml                               stage-1/3 model
  models/model.yaml                                                stage-2 model
  optim/adamw_cosine.yaml
data/
  process/               data pipeline (conversion, canonicalization, primitives, joint trajectories, frame cache)
  process/robomimic/     robomimic / MimicGen adapters, robosuite graphs and camera calibration
  dataset.py, data_config.py, embodiment_tags.py, transform/, collator.py, frame_cache_dataset.py
model/
  stage1.py, stage1_graph.py   motif tokenizer, motion-aware graph routing
  kinematic_graph.py           URDF -> kinematic graph
  stage2_3.py                  motif predictor and base policy
  stage3_elora.py, elora.py, geometry_tokens.py   E-LoRA policy and geometry tokens
  modules/                     transformer blocks, DiT flow-matching action head
src/
  train_stage_1_graph.py, train_stage_2.py, train_stage_3.py, train_utils.py
eval/
  run_percept_steering.py      ManiSkill benchmark driver (plain and steered)
  maniskill_spec.py/.json      task instructions, gripper columns and episode lengths of the ManiSkill data
  run_percept_steering_rs.py   robomimic benchmark driver
  robosuite_sim_server.py      robosuite simulator process used by the robomimic driver
  check_graph_nodes.py         sanity check of the kinematic graphs
experiment/
  inference_wrapper.py         checkpoint loading, (de)normalization, action decoding
scripts/
  setup_env.sh, check_setup.sh, vulkan_lavapipe_noroot.sh, maniskill-renderer.patch
  setup_robosuite_env.sh       second env for robomimic / MimicGen
  prepare_data.sh              ManiSkill data pipeline
  prepare_robomimic_data.sh    robomimic data pipeline
  train_chain.sh               ManiSkill: stages 1-3 + benchmark for one K
  run_experiment.sh            ManiSkill K sweep
  train_chain_robomimic.sh     robomimic: stages 1-3 + benchmark
```

## Requirements

| | |
| --- | --- |
| OS | Linux x86\_64 |
| Python / CUDA | Python 3.10.16, PyTorch 2.6.0 + CUDA 12.4, NVIDIA driver >= 550 |
| GPU | one GPU with >= 24 GB memory (about 15 GB used in stage 3 with batch 16); developed on an A100-80GB |
| CPU | the benchmarks run 3 simulator workers: >= 24 cores recommended |
| Disk | ManiSkill: replayed demos 12 GB, frame cache 11 GB, about 3 GB per training run. robomimic: renders about 31 GB, frame cache 8 GB |

Approximate time on one A100:

*   ManiSkill, per K: stage 1 6.5 h, stage 2 3 h, stage 3 15-17 h, Transfer benchmark 1.4 h plain + 1.1 h steered, Global benchmark 2.2 h + 2.2 h. Demonstration generation and replay take a few hours on 10 CPU workers.
*   robomimic: stage 1 about 3 h, stage 2 about 5 h, stage 3 about 1 day, benchmark under 1 h per run. Rendering and MimicGen generation take several hours on CPU.

## Environment setup

### Main environment

```bash
git clone <repo-url> && cd <repo>
bash scripts/setup_env.sh        # conda env, packages, ManiSkill fork + patch, robot assets, pretrained weights
conda activate kgraph
bash scripts/check_setup.sh      # verifies imports, the fork, assets, pretrained weights, GPU and rendering
```

`scripts/setup_env.sh` is safe to rerun. It does the following:

1.  Creates the conda env `kgraph` (Python 3.10.16), installs `torch==2.6.0 torchvision==0.21.0` from the cu124 index, then `pip install -r requirements.txt`.
2.  Clones the ManiSkill fork `https://github.com/Johnathan218/ManiSkill.git` at commit `52e1eb21dbe1fee29d18df223dc913d06b043481` into `../ManiSkill` (override with `MANISKILL_ROOT`). The fork is ManiSkill 3.0.0b21 plus the `*_wristcam` robots and motion-planning solutions for the xArm6 and WidowX AI. It applies `scripts/maniskill-renderer.patch`, which lets `MS_RENDER_BACKEND` select the renderer device, and installs the fork with `pip install --no-deps -e`.
3.  Downloads the xArm6 and WidowX AI robot assets into `~/.maniskill` (override with `MS_ASSET_DIR`).
4.  Caches the pretrained weights: `t5-base` (text encoder), `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` (geometry tokens) and `facebookresearch/dinov2` `dinov2_vitb14_reg` (image encoder). Set `SKIP_PRETRAINED=1` to skip. For an offline machine, copy `~/.cache/huggingface/hub` and `~/.cache/torch/hub` from a machine with internet access and set `HF_HUB_OFFLINE=1`.

Variables: `CONDA_ENV` (default `kgraph`), `MANISKILL_ROOT` (`../ManiSkill`), `MS_ASSET_DIR` (`~/.maniskill`), `MS_COMMIT`, `SKIP_PRETRAINED`.

### Software Vulkan renderer

ManiSkill demo replay and the ManiSkill benchmark render on the CPU with Mesa lavapipe, so the pixels do not depend on the GPU model.

*   **With root:** `apt install libvulkan1 mesa-vulkan-drivers`, then `export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json MS_RENDER_BACKEND=pci:0`.
*   **Without root:** `bash scripts/vulkan_lavapipe_noroot.sh <dir>` downloads and unpacks the Ubuntu 22.04 packages, then `export VK_FALLBACK_ROOT=<dir>/root`. `scripts/prepare_data.sh` and `eval/run_percept_steering.py` set `VK_ICD_FILENAMES`, `LD_LIBRARY_PATH` and `MS_RENDER_BACKEND=pci:0` from this variable.

A GPU Vulkan driver also works (leave both variables unset), but renders slightly different pixels.

### robosuite environment (robomimic benchmark)

MimicGen needs robosuite 1.4.1 and MuJoCo 2.3.2, which do not coexist with the main environment, so data generation and the robomimic simulator run in a second conda env:

```bash
bash scripts/setup_robosuite_env.sh
```

It creates the env `robosuite` (Python 3.10, OSMesa from conda-forge `mesalib=25.0.2`, CPU torch 2.3.1), installs `mujoco==2.3.2 robosuite==1.4.1 robomimic==0.3.0`, applies three small compatibility patches to robomimic 0.3.0 that MimicGen needs (optional `mujoco_py` import, `EnvRobosuite.base_env`, `env_class` argument of `create_env_for_data_processing`), clones MimicGen (`https://github.com/NVlabs/mimicgen`, commit `c0fbff22ba40060f05a917a80817f3f5c20731e7`) into `../mimicgen` and runs a headless render test of the four arms.

Variables: `SIM_ENV` (default `robosuite`), `MIMICGEN_ROOT` (`../mimicgen`), `MIMICGEN_COMMIT`. The robomimic benchmark driver starts the simulator with the python of this env; it finds it next to the main env (`<conda>/envs/robosuite/bin/python`), otherwise set `SIM_PYTHON`.

## Prepare the data

### ManiSkill

```bash
export MANISKILL_ROOT=$PWD/../ManiSkill
export VK_FALLBACK_ROOT=<dir>/root               # if you use the no-root renderer
bash scripts/prepare_data.sh                     # each step is skipped when its output exists
```

Everything is written under `demo_data/maniskill/`. Training reads `processed/` and `frame_cache/`; `build/` holds intermediate results and can be deleted at the end.

| Step | Script | Output |
| --- | --- | --- |
| 1\. generate | `$MANISKILL_ROOT/mani_skill/examples/motionplanning/<robot>/run.py --num-traj 50 --only-count-success --obs-mode none --sim-backend cpu` | `source/h5/<robot>/<task>/motionplanning/trajectory.h5`: 50 successful motion-planning demos per robot-task pair |
| 2\. replay | `$MANISKILL_ROOT/mani_skill/trajectory/replay_trajectory.py -o rgb -c pd_ee_delta_pose --save-traj -b physx_cpu` | `trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5`: 256x256 third-view + wrist images, joint positions, tcp pose, delta end-effector actions |
| 3\. convert | `data/process/maniskill_to_lerobot.py --fps 20 --base_camera third_view_camera --wrist_camera hand_camera --gripper_qpos_indices <cols> --gripper_norm dataset` | `lerobot/<robot>`: LeRobot v2.1 dataset; state = tcp pose (position, quaternion wxyz) + gripper |
| 4\. canonicalize | `data/process/trajectory_canonicalization.py --in_rotation_type quat --out_action_type rel --out_rotation_type quat` | `build/canonical/<robot>`: actions relative to the first frame, plus episode progress |
| 5\. segment | `data/process/segment_primitives.py --h5 demo_data/maniskill/source/h5` | `build/prim/segments.json`: rule-based motion segments |
| 6\. label | `data/process/label_primitives.py` | `build/prim/<robot>`: per-frame primitive labels, `PRIMITIVES_REPORT.json` |
| 7\. joint trajectories | `data/process/append_qpos.py --h5 demo_data/maniskill/source/h5` | `processed/<robot>`: adds `action.qpos` (joint positions in the joint order of `qpos_layout.json`), `build_stats.json` |
| 8\. frame cache | `data/process/prextract_frames.py --workers 12` | `frame_cache/<robot>`: decoded webp frames for stages 2 and 3, `PREXTRACT_DONE` |

Gripper columns of `obs/agent/qpos` used in step 3: Panda `7,8`, xArm6 `6`, WidowX AI `6,7`.

The script ends with a summary per robot: episode counts after each step, the action width and the joint-position columns of the processed dataset, and whether the frame cache is complete.

Variables: `RAW` (default `demo_data/maniskill/source/h5`), `NUM_TRAJ` (50), `NUM_PROCS` (10), `REPLAY_ENVS` (10), `ROBOTS`, `TASKS`, `CACHE_WORKERS` (12), `REBUILD_GRAPHS`.

To start from demos generated elsewhere, put them at `$RAW` in the same layout; steps 1-2 are then skipped. Motion planning is seeded per worker, but which plans succeed can depend on the host, so regenerated demos form a different (statistically equivalent) set.

### robomimic

Run from the main env (`conda activate kgraph`); the script switches to the robosuite env for the simulator steps:

```bash
bash scripts/prepare_robomimic_data.sh           # each step is skipped when its output exists
```

Everything is written under `demo_data/robomimic/`. Training reads `processed/` and `frame_cache/`.

| Step | Env | Script | Output |
| --- | --- | --- | --- |
| 1\. download | robosuite | `python -m robomimic.scripts.download_datasets --tasks lift can square --dataset_types ph --hdf5_types low_dim` | `source/robomimic/<task>_ph_low_dim.hdf5`: 200 Panda demos per task (MuJoCo states and actions) |
| 2\. render Panda | robosuite | `python -m robomimic.scripts.dataset_states_to_obs --camera_names agentview robot0_eye_in_hand --camera_height 256 --camera_width 256 --done_mode 2` | `source/robomimic/<task>_ph_256.hdf5` |
| 3\. MimicGen | robosuite | `data/process/robomimic/mimicgen_run.py prepare / generate --robot <arm> --n 100 --seed 1 / render` | `source/mimicgen/rs_<arm>_<task>_256.hdf5`: 100 successful demos per task and source arm |
| 4\. ManiSkill layout | main | `data/process/robomimic/to_maniskill_h5.py` (with images, and `--no-images` for the joint positions) | `source/h5/`, `source/h5_qpos/` |
| 5\. convert | main | `data/process/robomimic/pipeline.py to_lerobot` | `lerobot/<robot>`, `meta/gripper_range.json` (fixed gripper normalization, also used by the evaluator) |
| 6\. canonicalize | main | `data/process/trajectory_canonicalization.py` (as for ManiSkill) | `build/canonical/<robot>` |
| 7\. segment + label | main | `data/process/robomimic/pipeline.py segment`, `pipeline.py label` | `build/prim/` |
| 8\. joint trajectories | main | `data/process/robomimic/append_qpos_rs.py` | `processed/<robot>`: adds `action.qpos` (14 columns, zero-padded) |
| 9\. frame cache | main | `data/process/prextract_frames.py` | `frame_cache/<robot>` |

The robot keys are `rs_panda`, `rs_iiwa`, `rs_ur5e` and `rs_kinova3`. MimicGen ships no task configs for Lift and PickPlaceCan; `data/process/robomimic/mimicgen_ext.py` adds them (Square uses MimicGen's own). Generation is run until 100 demos succeed, so the number of attempts varies per task and arm.

Variables: `SIM_ENV` (`robosuite`), `MAIN_ENV` (the active env), `N_PANDA` (200), `N_GEN` (100), `PH_DIR` (use an existing robomimic download with `<task>/ph/low_dim*.hdf5` instead of downloading), `CACHE_WORKERS` (12).

### Real-robot data

The method reads more than the usual images, end-effector state, actions and instruction. The extra signals are joint positions for the motion-aware graph, object poses for the primitive labels, and a robot description plus wrist-camera calibration for the graph and the geometry tokens. In simulation these come from the simulator; on a real robot they must be recorded or calibrated.

**Recorded per frame, all at one fixed rate (20 Hz in the benchmarks):**

| Signal | Used by | Notes for a real robot |
| --- | --- | --- |
| Third-view and wrist RGB (256x256) | stages 2-3, steering | same as other policies |
| TCP pose in the robot base frame (position + quaternion wxyz) | relative actions, geometry tokens | compute it by forward kinematics from the same URDF as the kinematic graph, so every robot uses one TCP convention |
| Gripper opening | state and action | normalized with a fixed range per robot (`gripper_lo` / `gripper_hi`), which the evaluator applies too |
| Actions (delta end-effector pose + gripper command) | stage 3 | same as other policies |
| **Joint positions of every joint, gripper joints included** | stage 1 (`action.qpos`) | log the encoders in sync with the cameras; when the gripper reports only a width, convert it to the URDF joint angles, mimic joints included |
| **Object pose (position + quaternion wxyz) of the manipulated object** | primitive labels (`data/process/segment_primitives.py`) | the labelling rules use the object's distance to the TCP, whether it moved, whether it is lifted, its travel and its tilt; on hardware, use tracking (fiducial markers, motion capture or a pose estimator), or label the primitives by hand |
| Task instruction | stages 2-3 | the strings must match the K-shot table in the dataset config exactly |

**Measured once per robot and setup:**

| Item | File | Notes |
| --- | --- | --- |
| URDF with the gripper | `assets/kinematic_graphs/<robot>.json`, `qpos_layout.json` | build with `data/process/build_kinematic_graphs.py` and `build_qpos_layout.py`, then check sampled nodes with `eval/check_graph_nodes.py` |
| Wrist-camera intrinsics and hand-eye extrinsics | `assets/wrist_camera/*.json`: `intrinsic_cv`, `T_tcp_cam_cv`, `width`, `height` | from a hand-eye calibration; the intrinsics must match the recorded resolution; the pipeline crops (`crop_scale` 0.95) and resizes afterwards |
| Table height in the robot base frame | `plane_z` of the geometry tokens | the camera rays are intersected with this plane (0 in ManiSkill, 0.8 in robosuite) |
| Third-view camera intrinsics and pose | steering only | needed only when steering should fall back to the third view |

Depth comes from a monocular model (Depth-Anything-V2 metric), so no depth sensor is needed.

**Converting the recordings.** The robomimic adapter is the template: `data/process/robomimic/to_maniskill_h5.py` writes the ManiSkill trajectory layout, and the rest of the pipeline then runs unchanged. Per task, write `<root>/<robot>/<task>/motionplanning/trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5` with one group `traj_<i>` per episode, containing:

```plaintext
actions                          [T, action_dim]
obs/extra/tcp_pose               [T, 7]    position + quaternion wxyz, robot base frame
obs/extra/gripper_width          [T]       (or pass --gripper_qpos_indices for gripper columns of qpos)
obs/agent/qpos                   [T, dof]  joint order of qpos_layout.json
obs/sensor_data/<camera>/rgb     [T, 256, 256, 3] uint8, one group per camera
env_states/actors/<object>       [T, 7]    object pose, position + quaternion wxyz
success                          [T]       bool
```

The remaining steps follow `scripts/prepare_robomimic_data.sh`:

1.  Convert to LeRobot. Point `--base_camera` / `--wrist_camera` of `data/process/maniskill_to_lerobot.py` at the camera groups, and use `--gripper_norm fixed --gripper_lo --gripper_hi` with the gripper's physical range.
2.  Canonicalize with `data/process/trajectory_canonicalization.py`.
3.  Segment and label with `data/process/segment_primitives.py` and `label_primitives.py`. Set their task, object and gripper tables (`TASKS`, `OBJ`, `GRIP_OPEN`, `TASK_OF`) the way `data/process/robomimic/pipeline.py` does. For human teleoperation, also set `SPLIT_AT_GRIPPER = True`, because the arm keeps moving while the gripper closes.
4.  Append the joint trajectories and build the frame cache as in steps 7-8 above.

**Registering the robot:**

*   Add an embodiment tag with a new id in `data/embodiment_tags.py`. Keep `max_num_embodiments` in the model config above the largest id.
*   Add a data config with the suffixes `_rel`, `_rel_prim`, `_rel_prim_q` and `_rel_prim_geo` in `data/data_config.py`, following the `rs_*` entries.
*   Add a dataset yaml listing the processed folders and the K-shot table, following `config/dataset/dataset_robomimic_prim_q.yaml`.

**On the robot:**

*   `experiment/inference_wrapper.py` loads a stage-3 checkpoint, normalizes the observation with the checkpoint's `experiment_cfg/metadata.json` and decodes the action chunk. Always deploy with the `metadata.json` written by the training run.
*   The robot-side control loop is not part of this repository.
*   Perception steering is optional and training-free. It needs the object specs per task (colour, shape, size and interaction type, see `eval/run_percept_steering.py`) and a detector. The included detector uses fixed HSV colour masks; on hardware, tune the thresholds to the objects and lighting, or replace it.

### Kinematic graphs and camera calibration

The graphs in `assets/kinematic_graphs/` and the camera files in `assets/wrist_camera/` are included in the repository.

*   ManiSkill: `{panda,xarm6,widowxai}.json` are built from the robot URDFs (15 / 21 / 12 nodes); `qpos_layout.json` holds the joint order and limits (9 / 12 / 8 joints). To rebuild them from the fork's URDFs and compare with the included files:

    ```bash
    REBUILD_GRAPHS=1 bash scripts/prepare_data.sh    # writes demo_data/maniskill/build/graph_rebuild and prints identical / DIFFERS
    python eval/check_graph_nodes.py                 # samples nodes of each graph and checks them against the URDF
    ```

*   robosuite: `rs_{panda,iiwa,ur5e,kinova3}.json` are built from the compiled MuJoCo models, and their `qpos_layout.json` entries list `robot0_joint_pos` followed by `robot0_gripper_qpos`. `wrist_cameras_rs.json` holds the eye-in-hand mounts relative to the reported end-effector frame and the agentview camera of each task. To rebuild (robosuite env, `PYTHONPATH=$PWD`):

    ```bash
    python data/process/robomimic/build_kinematic_graphs.py --out <dir>
    python data/process/robomimic/build_qpos_layout_rs.py --graph_dir <dir> --layout <dir>/qpos_layout.json
    python data/process/robomimic/wrist_camera_calib.py --out <dir>/wrist_cameras_rs.json
    ```

## Training

### One command per K

ManiSkill:

```bash
K=5 GPU=0 bash scripts/train_chain.sh                                      # one K
KS="1 3 5 10 50" GPU=0 nohup bash scripts/run_experiment.sh >> runs/sweep.log 2>&1 &   # all K, one after another
```

`scripts/train_chain.sh` builds the frame cache (if missing), runs stages 1-3 and the Transfer benchmark (plain, then steered). `GLOBAL=1` adds the Global benchmark; `scripts/run_experiment.sh` sets it by default and prints a result table at the end.

robomimic:

```bash
K=50 GPU=0 bash scripts/train_chain_robomimic.sh
```

This runs the same three stages on the four robosuite arms (graphs `rs_*`, `wrist_cameras_rs.json`, table plane at z = 0.8, 16 embodiment slots), then the 450-episode benchmark, plain and steered.

Each stage resumes from its `last.pth` and is retried up to `MAX_TRIES` times, `WAIT` seconds apart. After an interruption, run the same command again; finished stages are skipped. A lock file allows one chain per run folder. `CHAIN_DONE` or `CHAIN_FAILED` in the run folder marks the end.

Variables of `scripts/train_chain.sh`: `K` (5), `GPU` (0), `OUT` (`runs/tgraph_k<k>`), `NAME` (`tgraph_k<k>`), `GLOBAL` (0), `MAX_TRIES` (8), `WAIT` (600), `CONDA_SH` + `CONDA_ENV` (to activate an env inside the script), `S1_EXTRA` / `S3_EXTRA` (extra Hydra overrides for stages 1 and 3). `scripts/train_chain_robomimic.sh` takes `K` (50), `GPU`, `OUT` (`runs/robomimic_k<k>`), `NAME` (`robomimic_k<k>`), `EPISODES` (150), `MAX_TRIES`, `WAIT`, `S1_EXTRA`, `S3_EXTRA`.

### Running the stages by hand

All commands run from the repository root with `PYTHONPATH=$PWD`. Training seed 42. The ManiSkill commands are shown; `scripts/train_chain_robomimic.sh` lists the robomimic overrides.

**Stage 1: motion-aware graph-routed motif tokenizer** (20 epochs, window 32, effective batch 8 x 16):

```bash
python src/train_stage_1_graph.py --config-name train_stage1_graph dataset=dataset_maniskill_prim_q K=5 seed=42 \
  output_dir=runs/tgraph_k5/outputs_stage1 train.epochs=20 train.grad_accum_steps=16 batch_size=8 \
  dataset.drop_video=true num_robots=3 tasks_num=6 \
  models.stage1.hard_training=true models.stage1.graph_temporal=true +train.resume_from=auto
```

**Stage 2: motif predictor** on the frozen tokenizer (30 epochs, effective batch 8 x 16):

```bash
python src/train_stage_2.py dataset=dataset_maniskill_prim_q +K=5 seed=42 output_dir=runs/tgraph_k5/outputs_stage2 \
  models.stage2.vq_vae_ckpt=runs/tgraph_k5/outputs_stage1/epoch_0020.pth \
  train.epochs=30 train.grad_accum_steps=16 batch_size=8 tasks_num=6 +train.resume_from=auto
```

**Stage 3: E-LoRA policy with geometry tokens** (60 epochs, window 16, effective batch 16 x 4):

```bash
python src/train_stage_3.py dataset=dataset_maniskill_prim_q dataset.data_config_suffix=_geo \
  models=model_graph_elora_geod +K=5 seed=42 output_dir=runs/tgraph_k5/outputs_stage3_elora_geod \
  models.stage2.vq_vae_ckpt=runs/tgraph_k5/outputs_stage1/epoch_0020.pth \
  models.stage3.mkl_ckpt=runs/tgraph_k5/outputs_stage2/epoch_0030.pth \
  train.epochs=60 train.grad_accum_steps=4 train.ckpt_interval=60 batch_size=16 dataset.num_workers=10 tasks_num=6 \
  models.stage1.graph_temporal=true +train.resume_from=auto
```

Stages 1-2 read the joint trajectories (`data_config_suffix=_q`, the dataset default); stage 3 reads the raw tcp pose for the geometry tokens (`_geo`). The stage-3 policy does not need joint positions at test time.

Main settings in `config/models/model_graph_elora_geod.yaml`: codebook size 128, model dim 256, graph MPNN with 3 layers and hidden size 128, temporal node features of size 32 (`graph_temporal_dim`), E-LoRA rank 4 on the DiT feed-forward layers, 16-layer DiT action head with 4 inference steps and a 16-action horizon.

## Evaluation

### ManiSkill: Transfer and Global

```bash
CKPT=runs/tgraph_k5/outputs_stage3_elora_geod/epoch_0060.pth
python eval/run_percept_steering.py --ckpt $CKPT --name tgraph_k5        --out eval_results/tgraph_k5        --steer 0
python eval/run_percept_steering.py --ckpt $CKPT --name tgraph_k5_steer  --out eval_results/tgraph_k5_steer  --steer 1.0
# Global: all 18 pairs; --reuse copies the 6 transfer pairs of a finished run with the same checkpoint and steering
python eval/run_percept_steering.py --ckpt $CKPT --name tgraph_k5_global --out eval_results/tgraph_k5_global --steer 0 \
  --all-pairs --reuse tgraph_k5
```

The driver copies `config.json` and links the checkpoint and its `experiment_cfg/` into `<out>/staged/`, then runs the pairs in `--workers` (default 3) simulator processes. Pairs that did not finish are retried (`--retries`, `--wait`). It writes one `<robot>__<task>.json` per pair, an episode journal and a per-step trace, and `<out>/SUMMARY.json` with the per-pair and total successes.

Useful options: `--episodes` (50), `--pairs 0 3` (subset of the pair list), `--policy-seed` (7), `--step-limit` (short smoke test). The first prediction of every pair is repeated to check that the per-request seed reproduces it exactly.

### robomimic

```bash
CKPT=runs/robomimic_k50/outputs_stage3_elora_geod/epoch_0060.pth
python eval/run_percept_steering_rs.py --ckpt $CKPT --name robomimic_k50       --out eval_results/robomimic_k50       --steer 0   --episodes 150
python eval/run_percept_steering_rs.py --ckpt $CKPT --name robomimic_k50_steer --out eval_results/robomimic_k50_steer --steer 1.0 --episodes 150
```

The policy runs in the main env; each worker starts `eval/robosuite_sim_server.py` in the robosuite env and talks to it over a unix socket. The simulator is built from the `env_args` of the robomimic hdf5 (`demo_data/robomimic/source/robomimic/<task>_ph_low_dim.hdf5`), with the arm swapped per pair; success is robosuite's `_check_success()`. The gripper state is normalized with `demo_data/robomimic/lerobot/<robot>/meta/gripper_range.json`. Options as above, plus `--all-pairs` (all four arms) and `--targets` (which pairs count as K-shot targets in the summary).

### Perception steering

With `--steer k > 0`, before each chunk is executed the target object is located from the robot's own cameras (colour segmentation, then a ray / known-height plane intersection from the calibrated wrist camera or the fixed third-view camera), and the chunk's planned xy endpoint is moved towards it with gain k over its first steps: the approach target while the gripper is open, the place target while it holds an object. Nothing is trained.

*   ManiSkill: grasp targets for PlaceSphere, StackCube and PickCube, the contact line for PushCube and PullCube, across the peg axis for LiftPegUpright.
*   robomimic: approach steering on Lift (handed over to the policy within 6 cm height / 2 cm of the object), the place servo towards the target bin on PickPlaceCan, no steering on NutAssemblySquare.

### Evaluating a downloaded checkpoint

The evaluators need the run folder of a checkpoint in this layout:

```plaintext
<run>/
  outputs_stage1/
    epoch_0020.pth, config.json         stage-1 tokenizer (its config defines the stage-1 architecture)
  outputs_stage2/
    epoch_0030.pth                      stage-2 motif predictor
  outputs_stage3_elora_geod/
    epoch_0060.pth                      stage-3 weights
    config.json                         model and dataset config of the run
    experiment_cfg/metadata.json        per-robot normalization statistics (min / max of states and actions)
```

`config.json` records the stage-1 and stage-2 checkpoints with paths from the training machine; when those paths do not exist, `experiment/inference_wrapper.py` looks for them in the run folder of the stage-3 checkpoint, so keep the three folders side by side.

`experiment/inference_wrapper.py` maps the policy's normalized outputs back to robot actions with `experiment_cfg/metadata.json`. Always use the `metadata.json` that ships with a checkpoint. Do not regenerate it from a locally built dataset: the statistics depend on the exact demonstrations, and statistics from different demonstrations make the policy send wrong actions. The benchmarks run in simulation and read no dataset.

```bash
python eval/run_percept_steering.py --ckpt <download>/outputs_stage3_elora_geod/epoch_0060.pth \
  --name released_k5 --out eval_results/released_k5 --steer 0
```

## Output directories

```plaintext
runs/tgraph_k<k>/                  ManiSkill run (runs/robomimic_k<k>/ for robomimic)
  outputs_stage1/epoch_0020.pth            stage-1 tokenizer
  outputs_stage2/epoch_0030.pth            stage-2 motif predictor
  outputs_stage3_elora_geod/
    epoch_0060.pth, config.json, experiment_cfg/metadata.json
  stage1.log, stage2.log, stage3.log, eval.log, ...   per-stage logs of the chain script
  CHAIN_DONE | CHAIN_FAILED
eval_results/
  tgraph_k<k>/                 Transfer, plain
  tgraph_k<k>_steer/           Transfer, steered
  tgraph_k<k>_global/          Global, plain
  tgraph_k<k>_global_steer/    Global, steered
  robomimic_k<k>/, robomimic_k<k>_steer/
    SUMMARY.json, <name>/<robot>__<task>.json, logs/
```

`last.pth` files are removed when the chain finishes.

## Reproducibility notes

Fixed: the data pipelines, the K-shot selection, all seeds (training 42, environments 2025 onwards, policy 7 with per-request SHA-256 seeds, MimicGen seed 1), epochs, batch size x gradient accumulation, window sizes, optimizer settings (`config/optim/adamw_cosine.yaml`), the benchmark protocols, the software renderer, the pinned package versions (`requirements.txt`, `scripts/setup_robosuite_env.sh`) and the ManiSkill and MimicGen commits.

Not bit-exact:

*   Regenerated demonstrations: motion-planning and MimicGen success depend on the host, so the set of accepted demos can differ.
*   GPU kernels: some CUDA operations (cuDNN, `index_add`) are nondeterministic.
*   Renderer library versions: lavapipe from Ubuntu 22.04 (Mesa 23.2.1) for ManiSkill, OSMesa from conda-forge `mesalib` 25.0.2 for robosuite.

Expect results within the usual seed-to-seed spread, not identical rollouts.

## Acknowledgements

Parts of this code are derived from MIT-licensed code (see `LICENSE`) and from the GR00T data and action-head code (NVIDIA, Apache-2.0; see the headers of the respective files). We also thank the authors of UniVLA, [ManiSkill 3](https://github.com/haosulab/ManiSkill), [robosuite](https://github.com/ARISE-Initiative/robosuite), [robomimic](https://github.com/ARISE-Initiative/robomimic) and [MimicGen](https://github.com/NVlabs/mimicgen).
