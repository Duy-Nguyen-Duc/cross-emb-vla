#!/bin/bash
# Conda env, pinned packages, ManiSkill fork + renderer patch, robot assets, pretrained weights. Safe to rerun.
#   bash scripts/setup_env.sh
# Variables: CONDA_ENV (kgraph), MANISKILL_ROOT (../ManiSkill), MS_ASSET_DIR (~/.maniskill), MS_COMMIT, SKIP_PRETRAINED=1
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
CONDA_ENV=${CONDA_ENV:-kgraph}
MANISKILL_ROOT=${MANISKILL_ROOT:-$ROOT/../ManiSkill}
MS_COMMIT=${MS_COMMIT:-52e1eb21dbe1fee29d18df223dc913d06b043481}
PATCH=$ROOT/scripts/maniskill-renderer.patch
log() { echo "[$(date -u '+%F %T')] setup $*"; }

command -v conda > /dev/null || { echo "conda not found: install Miniconda first"; exit 1; }
eval "$(conda shell.bash hook)"
conda env list | grep -qE "^$CONDA_ENV\s" || { log "creating env $CONDA_ENV"; conda create -y -n "$CONDA_ENV" python=3.10.16 pip; }
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1
python -c "import sys; assert sys.version_info[:2] == (3, 10), sys.version"

log "torch 2.6.0 + cu124"
python -m pip install --quiet torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
log "pinned requirements"
python -m pip install --quiet -r requirements.txt

log "ManiSkill fork at $MANISKILL_ROOT ($MS_COMMIT)"
if [ ! -d "$MANISKILL_ROOT/.git" ]; then
  git clone https://github.com/Johnathan218/ManiSkill.git "$MANISKILL_ROOT"
fi
git -C "$MANISKILL_ROOT" checkout -q "$MS_COMMIT"
if grep -q MS_RENDER_BACKEND "$MANISKILL_ROOT/mani_skill/envs/utils/system/backend.py"; then
  log "renderer patch already applied"
elif git -C "$MANISKILL_ROOT" apply --check "$PATCH" 2> /dev/null; then
  git -C "$MANISKILL_ROOT" apply "$PATCH"; log "renderer patch applied"
else
  echo "renderer patch does not apply to $MANISKILL_ROOT (wrong commit or local edits)"; exit 1
fi
python -m pip install --quiet --no-deps -e "$MANISKILL_ROOT"
python -c "import mani_skill, os; assert os.path.realpath(mani_skill.__file__).startswith(os.path.realpath('$MANISKILL_ROOT')), mani_skill.__file__"

log "robot assets (xarm6, widowxai) into ${MS_ASSET_DIR:-~/.maniskill}"
for a in xarm6 widowxai; do python -m mani_skill.utils.download_asset -y "$a"; done

if [ "${SKIP_PRETRAINED:-0}" != 1 ]; then
  log "pretrained caches: t5-base, Depth-Anything-V2-Metric-Indoor-Small, dinov2_vitb14_reg"
  python - <<'PY'
from transformers import T5EncoderModel, T5Tokenizer, AutoModelForDepthEstimation
import torch
T5EncoderModel.from_pretrained("t5-base"); T5Tokenizer.from_pretrained("t5-base")
AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
print("pretrained caches OK")
PY
fi
log "done; next: bash scripts/check_setup.sh"
