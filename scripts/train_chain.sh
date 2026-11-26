#!/bin/bash
# Full pipeline for one K: frame cache -> stage 1 (motion-aware graph-routed motif tokenizer) -> stage 2 (motif
# predictor) -> stage 3 (E-LoRA policy + geometry tokens) -> Transfer benchmark, plain and steered.
# Every stage resumes from last.pth and is retried; rerun the same command after an interruption.
#   K=5 GPU=0 bash scripts/train_chain.sh
# Variables: K (5), GPU (0), OUT (runs/tgraph_k<K>), NAME (tgraph_k<K>), GLOBAL=1 (also all 18 pairs),
#   MAX_TRIES (8), WAIT (600 s between retries), CONDA_SH / CONDA_ENV (activate an env first; default: current env),
#   S1_EXTRA / S3_EXTRA (extra Hydra overrides for stages 1 and 3).
set -u
K=${K:-5}; GPU=${GPU:-0}; SEED=42; MAX_TRIES=${MAX_TRIES:-8}; WAIT=${WAIT:-600}; GLOBAL=${GLOBAL:-0}; CKPT_EVERY=${CKPT_EVERY:-60}
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
OUT=${OUT:-runs/tgraph_k${K}}; NAME=${NAME:-tgraph_k${K}}; CACHE=demo_data/maniskill/frame_cache
S1_EXTRA=${S1_EXTRA:-}; S3_EXTRA=${S3_EXTRA:-}; DS=dataset_maniskill_prim_q; TEMPORAL=models.stage1.graph_temporal=true
S1=$OUT/outputs_stage1; S2=$OUT/outputs_stage2; S3=$OUT/outputs_stage3_elora_geod
mkdir -p "$OUT"
exec 9> "$OUT/.lock"   # one chain per run folder
flock -n 9 || { echo "[$(date -u '+%F %T')] $OUT: this chain is already running, nothing launched"; exit 0; }
CONDA_SH=${CONDA_SH:-}; CONDA_ENV=${CONDA_ENV:-kgraph}
if [ -n "$CONDA_SH" ] && [ -f "$CONDA_SH" ]; then source "$CONDA_SH" && conda activate "$CONDA_ENV"; fi   # else: launched inside an activated env
export CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT"
log() { echo "[$(date -u '+%F %T')] K=$K gpu=$GPU $*"; }
run_stage() {   # label artifact cmd...: rerun until the artifact exists
  local label=$1 art=$2 attempt rc; shift 2
  for attempt in $(seq 1 "$MAX_TRIES"); do
    [ -f "$art" ] && { log "$label: $art exists"; return 0; }
    log "$label attempt $attempt"
    "$@" >> "$OUT/$label.log" 2>&1 9>&-; rc=$?
    [ -f "$art" ] && { log "$label done (rc=$rc)"; return 0; }
    log "$label failed rc=$rc; retry in $WAIT s"; sleep "$WAIT"
  done
  log "$label FAILED after $MAX_TRIES attempts"; touch "$OUT/CHAIN_FAILED"; exit 1
}

# stage 0: webp cache of every frame (read by stages 2 and 3)
[ -f "$CACHE/PREXTRACT_DONE" ] || { python data/process/prextract_frames.py --dataset_root demo_data/maniskill/processed \
    --out_root "$CACHE" --robots panda xarm6 widowxai --format webp --workers 12 9>&- && touch "$CACHE/PREXTRACT_DONE"; } \
    || { log "frame cache FAILED"; touch "$OUT/CHAIN_FAILED"; exit 1; }

# stage 1: motion-aware graph-routed motif tokenizer, hard routing
run_stage stage1 "$S1/epoch_0020.pth" \
  python src/train_stage_1_graph.py --config-name train_stage1_graph dataset=$DS K=$K seed=$SEED output_dir="$S1" \
    train.epochs=20 train.grad_accum_steps=16 dataset.drop_video=true +train.resume_from=auto \
    batch_size=8 num_robots=3 tasks_num=6 models.stage1.hard_training=true $TEMPORAL $S1_EXTRA

# stage 2: multimodal motif predictor on the frozen tokenizer
run_stage stage2 "$S2/epoch_0030.pth" \
  python src/train_stage_2.py dataset=$DS +K=$K seed=$SEED output_dir="$S2" \
    models.stage2.vq_vae_ckpt="$S1/epoch_0020.pth" train.epochs=30 train.grad_accum_steps=16 \
    +train.resume_from=auto batch_size=8 tasks_num=6

# stage 3: graph-routed E-LoRA policy with geometry tokens
run_stage stage3 "$S3/epoch_0060.pth" \
  python src/train_stage_3.py dataset=$DS dataset.data_config_suffix=_geo \
    models=model_graph_elora_geod +K=$K seed=$SEED output_dir="$S3" \
    models.stage2.vq_vae_ckpt="$S1/epoch_0020.pth" models.stage3.mkl_ckpt="$S2/epoch_0030.pth" \
    train.epochs=60 train.grad_accum_steps=4 train.ckpt_interval=$CKPT_EVERY +train.resume_from=auto \
    batch_size=16 dataset.num_workers=10 tasks_num=6 $TEMPORAL $S3_EXTRA

# Transfer benchmark (6 pairs x 50 episodes): policy alone, then with perception steering
run_stage eval "eval_results/$NAME/SUMMARY.json" \
  python eval/run_percept_steering.py --ckpt "$S3/epoch_0060.pth" --name $NAME --out eval_results/$NAME --steer 0
run_stage steer "eval_results/${NAME}_steer/SUMMARY.json" \
  python eval/run_percept_steering.py --ckpt "$S3/epoch_0060.pth" --name ${NAME}_steer --out eval_results/${NAME}_steer --steer 1.0

# Global benchmark (18 pairs); the 6 transfer pairs are copied from the runs above
if [ "$GLOBAL" = 1 ]; then
  run_stage global "eval_results/${NAME}_global/SUMMARY.json" \
    python eval/run_percept_steering.py --ckpt "$S3/epoch_0060.pth" --name ${NAME}_global --out eval_results/${NAME}_global --steer 0 --all-pairs --reuse $NAME
  run_stage global_steer "eval_results/${NAME}_global_steer/SUMMARY.json" \
    python eval/run_percept_steering.py --ckpt "$S3/epoch_0060.pth" --name ${NAME}_global_steer --out eval_results/${NAME}_global_steer --steer 1.0 --all-pairs --reuse ${NAME}_steer
fi

rm -f "$S3/last.pth" "$S2/last.pth" "$S1/last.pth"
log "ALL DONE"; touch "$OUT/CHAIN_DONE"
