#!/bin/bash
# Runs scripts/train_chain.sh for each K in turn on one GPU, then prints the results.
#   K=5 GPU=0 bash scripts/run_experiment.sh
#   KS="1 3 5 10 50" GPU=0 nohup bash scripts/run_experiment.sh >> runs/sweep.log 2>&1 &
# Variables: KS or K (5), GPU (0), GLOBAL (1), MAX_TRIES (8), WAIT (600 s)
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
KS=${KS:-${K:-5}}; GPU=${GPU:-0}
[ -f demo_data/maniskill/processed/build_stats.json ] || { echo "no demo_data/maniskill/processed: run scripts/prepare_data.sh first"; exit 1; }
export MANISKILL_ROOT=${MANISKILL_ROOT:-$ROOT/../ManiSkill}
log() { echo "[$(date -u '+%F %T')] $*"; }
for K in $KS; do
  O=runs/tgraph_k$K
  mkdir -p "$O"
  log "K=$K -> $O"
  K=$K GPU=$GPU OUT=$O NAME=tgraph_k$K GLOBAL=${GLOBAL:-1} bash scripts/train_chain.sh >> "$O/chain.log" 2>&1
  [ -f "$O/CHAIN_DONE" ] || { log "K=$K did not finish (see $O/chain.log); rerun this command to resume"; exit 1; }
done
python - $KS <<'PY'
import json, os, sys
print(f"{'K':>3} {'Transfer':>9} {'+steer':>7} {'Global':>8} {'+steer':>7}   (successes of 300 / 300 / 900 / 900)")
for k in sys.argv[1:]:
    cells = []
    for sfx in ("", "_steer", "_global", "_global_steer"):
        p = f"eval_results/tgraph_k{k}{sfx}/SUMMARY.json"
        cells.append(str(json.load(open(p))["successes"]) if os.path.exists(p) else "-")
    print(f"{k:>3} {cells[0]:>9} {cells[1]:>7} {cells[2]:>8} {cells[3]:>7}")
PY
