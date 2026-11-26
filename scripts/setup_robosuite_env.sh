#!/bin/bash
# Second conda env for the robomimic benchmark: robosuite 1.4.1, robomimic 0.3.0, MuJoCo 2.3.2, MimicGen, OSMesa
# rendering, CPU torch. Data generation and the simulator run here; training and the policy run in the main env.
#   bash scripts/setup_robosuite_env.sh
# Variables: SIM_ENV (robosuite), MIMICGEN_ROOT (../mimicgen), MIMICGEN_COMMIT
set -eu -o pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
SIM_ENV=${SIM_ENV:-robosuite}
MIMICGEN_ROOT=${MIMICGEN_ROOT:-$ROOT/../mimicgen}
MIMICGEN_COMMIT=${MIMICGEN_COMMIT:-c0fbff22ba40060f05a917a80817f3f5c20731e7}
log() { echo "[$(date -u '+%F %T')] setup $*"; }

command -v conda > /dev/null || { echo "conda not found: install Miniconda first"; exit 1; }
eval "$(conda shell.bash hook)"
conda env list | grep -qE "^$SIM_ENV\s" || { log "creating env $SIM_ENV"; conda create -y -n "$SIM_ENV" -c conda-forge python=3.10 "mesalib=25.0.2" glew numpy=1.26 pip; }
conda activate "$SIM_ENV"
export PYTHONNOUSERSITE=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH="$CONDA_PREFIX/lib"

log "cpu torch"
python -c "import torch; assert '+cpu' in torch.__version__" 2> /dev/null || \
  python -m pip install --quiet "torch==2.3.1" "torchvision==0.18.1" --index-url https://download.pytorch.org/whl/cpu
log "mujoco, robosuite, robomimic"
python -m pip install --quiet "mujoco==2.3.2" "robosuite==1.4.1" "robomimic==0.3.0" "numpy<2" h5py imageio imageio-ffmpeg scipy

log "robomimic 0.3.0 patches (optional mujoco_py, base_env, env_class argument)"
python - <<'PY'
from pathlib import Path
import robomimic
f = Path(robomimic.__file__).parent / "envs/env_robosuite.py"
s = f.read_text()
if "mujoco_py = None" not in s:
    s = s.replace("import mujoco_py\n", "try:\n    import mujoco_py\nexcept ImportError:\n    mujoco_py = None\n", 1)
s = s.replace("        return (mujoco_py.builder.MujocoException)\n", "        return () if mujoco_py is None else (mujoco_py.builder.MujocoException,)\n", 1)
if "def base_env" not in s:
    s = s.replace("    def get_state(self):\n", "    @property\n    def base_env(self):\n        return self.env\n\n    def get_state(self):\n", 1)
f.write_text(s)
g = Path(robomimic.__file__).parent / "utils/env_utils.py"
u = g.read_text()
if "env_class=None" not in u:
    u = u.replace("    reward_shaping,\n):\n", "    reward_shaping,\n    env_class=None, render=None, render_offscreen=None, use_image_obs=None, use_depth_obs=None,\n):\n", 1)
    u = u.replace("    env_class = get_env_class(env_type=env_type)\n", "    if env_class is None:\n        env_class = get_env_class(env_type=env_type)\n", 1)
    g.write_text(u)
print("robomimic patches ok")
PY

log "MimicGen at $MIMICGEN_ROOT ($MIMICGEN_COMMIT)"
[ -d "$MIMICGEN_ROOT/.git" ] || git clone https://github.com/NVlabs/mimicgen.git "$MIMICGEN_ROOT"
git -C "$MIMICGEN_ROOT" checkout -q "$MIMICGEN_COMMIT"
python -m pip install --quiet -e "$MIMICGEN_ROOT"

log "headless render test"
python - <<'PY'
import numpy as np
import robosuite
from robosuite.controllers import load_controller_config
for robot in ["Panda", "IIWA", "UR5e", "Kinova3"]:
    env = robosuite.make("Lift", robots=robot, controller_configs=load_controller_config(default_controller="OSC_POSE"),
                         has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
                         camera_names=["agentview", "robot0_eye_in_hand"], camera_heights=256, camera_widths=256)
    obs = env.reset()
    obs, *_ = env.step(np.zeros(7))
    print(robot, obs["agentview_image"].shape, round(float(obs["agentview_image"].mean()), 1))
    env.close()
import mimicgen  # noqa: F401
print("RENDER OK")
PY
log "done"
