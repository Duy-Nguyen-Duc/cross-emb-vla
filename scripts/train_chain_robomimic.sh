#!/bin/bash
# robomimic benchmark (Panda target with K demos per task; IIWA, UR5e, Kinova3 sources): stage 1 -> stage 2 -> stage 3 ->
# 3 Panda tasks x EPISODES episodes, plain and steered. Resumable like scripts/train_chain.sh.
#   K=50 GPU=0 bash scripts/train_chain_robomimic.sh
# Variables: K (50), GPU (0), OUT (runs/robomimic_k<K>), NAME (robomimic_k<K>), EPISODES (150), MAX_TRIES (8), WAIT (600 s),
#   SIM_PYTHON (python of the robosuite env), S1_EXTRA / S3_EXTRA (extra Hydra overrides for stages 1 and 3).
set -u
K=${K:-50}; GPU=${GPU:-0}; SEED=42; MAX_TRIES=${MAX_TRIES:-8}; WAIT=${WAIT:-600}; EPISODES=${EPISODES:-150}
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
OUT=${OUT:-runs/robomimic_k${K}}; NAME=${NAME:-robomimic_k${K}}; CACHE=demo_data/robomimic/frame_cache
S1_EXTRA=${S1_EXTRA:-}; S3_EXTRA=${S3_EXTRA:-}; DS=dataset_robomimic_prim_q; TEMPORAL=models.stage1.graph_temporal=true
S1=$OUT/outputs_stage1; S2=$OUT/outputs_stage2; S3=$OUT/outputs_stage3_elora_geod
ROBOTS='{rs_panda:rs_panda_rel,rs_iiwa:rs_iiwa_rel,rs_ur5e:rs_ur5e_rel,rs_kinova3:rs_kinova3_rel}'
GEO="models.stage3.geometry_cfg.calib=assets/wrist_camera/wrist_cameras_rs.json +models.stage3.geometry_cfg.plane_z=0.8"
mkdir -p "$OUT"
exec 9> "$OUT/.lock"
flock -n 9 || { echo "[$(date -u '+%F %T')] $OUT: this chain is already running, nothing launched"; exit 0; }
export CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT"
log() { echo "[$(date -u '+%F %T')] K=$K gpu=$GPU $*"; }
run_stage() {
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

[ -f "$CACHE/PREXTRACT_DONE" ] || { python data/process/prextract_frames.py --dataset_root demo_data/robomimic/processed \
    --out_root "$CACHE" --robots rs_panda rs_iiwa rs_ur5e rs_kinova3 --format webp --workers 12 9>&- && touch "$CACHE/PREXTRACT_DONE"; } \
    || { log "frame cache FAILED"; touch "$OUT/CHAIN_FAILED"; exit 1; }

run_stage stage1 "$S1/epoch_0020.pth" \
  python src/train_stage_1_graph.py --config-name train_stage1_graph dataset=$DS K=$K seed=$SEED output_dir="$S1" \
    train.epochs=20 train.grad_accum_steps=16 dataset.drop_video=true +train.resume_from=auto \
    batch_size=8 num_robots=4 tasks_num=3 models.stage1.hard_training=true "~models.stage1.robots" "+models.stage1.robots=$ROBOTS" \
    $TEMPORAL $S1_EXTRA

run_stage stage2 "$S2/epoch_0030.pth" \
  python src/train_stage_2.py dataset=$DS +K=$K seed=$SEED output_dir="$S2" \
    models.stage2.vq_vae_ckpt="$S1/epoch_0020.pth" train.epochs=30 train.grad_accum_steps=16 \
    +train.resume_from=auto batch_size=8 tasks_num=3

# embodiment ids 10-13 -> 16 projector slots
run_stage stage3 "$S3/epoch_0060.pth" \
  python src/train_stage_3.py dataset=$DS dataset.data_config_suffix=_geo \
    models=model_graph_elora_geod +K=$K seed=$SEED output_dir="$S3" "~models.stage1.robots" "+models.stage1.robots=$ROBOTS" $GEO \
    models.stage3.flow_matching_cfg.max_num_embodiments=16 \
    models.stage2.vq_vae_ckpt="$S1/epoch_0020.pth" models.stage3.mkl_ckpt="$S2/epoch_0030.pth" \
    train.epochs=60 train.grad_accum_steps=4 train.ckpt_interval=60 +train.resume_from=auto \
    batch_size=16 dataset.num_workers=10 tasks_num=3 $TEMPORAL $S3_EXTRA

run_stage eval "eval_results/$NAME/SUMMARY.json" \
  python eval/run_percept_steering_rs.py --ckpt "$S3/epoch_0060.pth" --name $NAME --out eval_results/$NAME --steer 0 --episodes $EPISODES
run_stage steer "eval_results/${NAME}_steer/SUMMARY.json" \
  python eval/run_percept_steering_rs.py --ckpt "$S3/epoch_0060.pth" --name ${NAME}_steer --out eval_results/${NAME}_steer --steer 1.0 --episodes $EPISODES

rm -f "$S3/last.pth" "$S2/last.pth" "$S1/last.pth"
log "ALL DONE"; touch "$OUT/CHAIN_DONE"
