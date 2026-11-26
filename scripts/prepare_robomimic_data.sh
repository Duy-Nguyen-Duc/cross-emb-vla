#!/bin/bash
# Builds the robomimic training data; every step is skipped when its output exists:
#   download Panda PH demos -> render (256x256, both cameras) -> MimicGen source-arm demos (IIWA, UR5e, Kinova3) ->
#   ManiSkill layout -> LeRobot -> canonical rel-state actions -> primitive labels -> joint trajectories -> frame cache.
#   bash scripts/prepare_robomimic_data.sh
# Variables: SIM_ENV (robosuite), MAIN_ENV (current env), N_PANDA (200 demos per task), N_GEN (100 per task and arm),
#   PH_DIR (an existing robomimic download with <task>/ph/low_dim*.hdf5; default: download), CACHE_WORKERS (12)
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
SIM_ENV=${SIM_ENV:-robosuite}; MAIN_ENV=${MAIN_ENV:-${CONDA_DEFAULT_ENV:-}}
N_PANDA=${N_PANDA:-200}; N_GEN=${N_GEN:-100}
SRC=demo_data/robomimic/source; PH=$SRC/robomimic; MG=$SRC/mimicgen
TASKS="Lift PickPlaceCan NutAssemblySquare"; ARMS="IIWA UR5e Kinova3"
ROBOTS="rs_panda rs_iiwa rs_ur5e rs_kinova3"
declare -A SHORT=([Lift]=lift [PickPlaceCan]=can [NutAssemblySquare]=square)
declare -A KEY=([Panda]=rs_panda [IIWA]=rs_iiwa [UR5e]=rs_ur5e [Kinova3]=rs_kinova3)
log() { echo "[$(date -u '+%F %T')] data $*"; }
eval "$(conda shell.bash hook)"
mkdir -p "$PH" "$MG" runs/data_logs
export PYTHONNOUSERSITE=1 PYTHONPATH=$ROOT

LD0=${LD_LIBRARY_PATH:-}
sim_env() {
  conda activate "$SIM_ENV"
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=$CONDA_PREFIX/lib OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
}
sim_env

for t in $TASKS; do
  s=${SHORT[$t]}
  [ -f "$PH/${s}_ph_low_dim.hdf5" ] && continue
  if [ -z "${PH_DIR:-}" ]; then
    PH_DIR=$SRC/download
    log "download robomimic PH low-dim demos -> $PH_DIR"
    python -m robomimic.scripts.download_datasets --tasks lift can square --dataset_types ph --hdf5_types low_dim --download_dir "$PH_DIR"
  fi
  cp "$(ls "$PH_DIR/$s/ph/"low_dim*.hdf5 | head -1)" "$PH/${s}_ph_low_dim.hdf5"
done

for t in $TASKS; do
  s=${SHORT[$t]}
  [ -f "$PH/${s}_ph_256.hdf5" ] && continue
  log "render Panda $t ($N_PANDA demos)"
  python -m robomimic.scripts.dataset_states_to_obs --dataset "$PH/${s}_ph_low_dim.hdf5" --output_name "${s}_ph_256.hdf5" \
    --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 256 --camera_width 256 --n "$N_PANDA" --compress \
    > "runs/data_logs/render_panda_$s.log" 2>&1 &
done
wait

for t in $TASKS; do
  [ -f "$MG/src/${SHORT[$t]}_src.hdf5" ] || { log "MimicGen source annotation $t"; python data/process/robomimic/mimicgen_run.py prepare --task "$t" --src-demos "$N_PANDA"; }
done

pids=""
for t in $TASKS; do for r in $ARMS; do
  [ -f "$MG/gen/${SHORT[$t]}_$r/demo.hdf5" ] && continue
  log "MimicGen generate $t / $r ($N_GEN successful demos)"
  python data/process/robomimic/mimicgen_run.py generate --task "$t" --robot "$r" --n "$N_GEN" --src-demos "$N_PANDA" --seed 1 > "runs/data_logs/gen_${SHORT[$t]}_$r.log" 2>&1 &
  pids="$pids $!"
done; done
for p in $pids; do wait "$p"; done
pids=""
for t in $TASKS; do for r in $ARMS; do
  [ -f "$MG/${KEY[$r]}_${SHORT[$t]}_256.hdf5" ] && continue
  log "render $t / $r"
  python data/process/robomimic/mimicgen_run.py render --task "$t" --robot "$r" --n "$N_GEN" > "runs/data_logs/render_${SHORT[$t]}_$r.log" 2>&1 &
  pids="$pids $!"
done; done
for p in $pids; do wait "$p"; done
for t in $TASKS; do
  [ -f "$PH/${SHORT[$t]}_ph_256.hdf5" ] || { echo "missing $PH/${SHORT[$t]}_ph_256.hdf5"; exit 1; }
  for r in $ARMS; do [ -f "$MG/${KEY[$r]}_${SHORT[$t]}_256.hdf5" ] || { echo "missing $MG/${KEY[$r]}_${SHORT[$t]}_256.hdf5"; exit 1; }; done
done

conda activate "$MAIN_ENV"
unset MUJOCO_GL PYOPENGL_PLATFORM; export LD_LIBRARY_PATH=$LD0
for r in $ROBOTS; do
  for t in $TASKS; do
    s=${SHORT[$t]}
    [ "$r" = rs_panda ] && src=$PH/${s}_ph_256.hdf5 || src=$MG/${r}_${s}_256.hdf5
    [ -f "$SRC/h5/$r/$t/motionplanning/trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5" ] || \
      python data/process/robomimic/to_maniskill_h5.py --src "$src" --robot "$r" --task "$t"
    [ -f "$SRC/h5_qpos/$r/$t/motionplanning/trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5" ] || \
      python data/process/robomimic/to_maniskill_h5.py --src "$src" --robot "$r" --task "$t" --no-images --out "$SRC/h5_qpos"
  done
  [ -f "demo_data/robomimic/lerobot/$r/meta/episodes.jsonl" ] || { log "convert $r -> demo_data/robomimic/lerobot/$r"; python data/process/robomimic/pipeline.py to_lerobot --robot "$r"; }
done

done_all=1; for r in $ROBOTS; do [ -f "demo_data/robomimic/build/canonical/$r/meta/episodes.jsonl" ] || done_all=0; done
if [ $done_all = 0 ]; then
  log "canonicalize -> demo_data/robomimic/build/canonical"
  python data/process/trajectory_canonicalization.py --dataset_path demo_data/robomimic/lerobot --save_path demo_data/robomimic/build/canonical \
    --in_rotation_type quat --out_action_type rel --out_rotation_type quat
fi
[ -f demo_data/robomimic/build/prim/segments.json ] || { log "segment primitives"; python data/process/robomimic/pipeline.py segment --robots $ROBOTS; }
[ -f demo_data/robomimic/build/prim/PRIMITIVES_REPORT.json ] || { log "label primitives"; python data/process/robomimic/pipeline.py label --robots $ROBOTS; }
[ -f demo_data/robomimic/processed/build_stats.json ] || { log "append joint trajectories -> demo_data/robomimic/processed"; python data/process/robomimic/append_qpos_rs.py --robots $ROBOTS; }

if [ ! -f demo_data/robomimic/frame_cache/PREXTRACT_DONE ]; then
  log "frame cache -> demo_data/robomimic/frame_cache (webp)"
  python data/process/prextract_frames.py --dataset_root demo_data/robomimic/processed --out_root demo_data/robomimic/frame_cache \
    --robots $ROBOTS --format webp --workers "${CACHE_WORKERS:-12}" && touch demo_data/robomimic/frame_cache/PREXTRACT_DONE
fi

python - $ROBOTS <<'PY'
import json, sys
for r in sys.argv[1:]:
    n = sum(1 for _ in open(f"demo_data/robomimic/processed/{r}/meta/episodes.jsonl"))
    print(f"{r}: {n} episodes")
PY
log "DATA READY"
