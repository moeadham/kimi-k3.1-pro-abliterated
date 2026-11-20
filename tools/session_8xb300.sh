#!/usr/bin/env bash
# Step-2 full-model session on 8xB300 (docs/plan_step2.md §2.2). Resumable: each stage
# writes $OUT/<stage>/DONE and is skipped on rerun.
#
#   MODEL=/local/Kimi-K3.1-Pro OUT=/local/session [WEIGHTS_FROM=/workspace/Kimi-K3.1-Pro] \
#     bash tools/session_8xb300.sh [stage ...]
#
# Needs: tools/setup_pod.sh done; ALLMODELS_API_KEY (the Jev judge) in the environment (read it from stdin:
#   read -rs ALLMODELS_API_KEY; export ALLMODELS_API_KEY) -- never on the command line or disk.
# The `bake` stage needs $OUT/tier2/winner.json, chosen by a human from $OUT/tier2/report.json.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-/local/Kimi-K3.1-Pro}
OUT=${OUT:-/local/session}
TP=${TP:-8}
mkdir -p "$OUT"
: "${ALLMODELS_API_KEY:?set ALLMODELS_API_KEY (read -rs) before running}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee "$OUT/gpus.txt"
df -h "$(dirname "$MODEL")" | tee "$OUT/disk.txt"

if [ -n "${WEIGHTS_FROM:-}" ]; then
  python tools/download_weights.py "$MODEL" --from-dir "$WEIGHTS_FROM"
else
  python tools/download_weights.py "$MODEL"
fi

STAGES=${*:-checks capture fidelity directions baseline sweep tier2}
for st in $STAGES; do
  echo "=== stage $st ===" | tee -a "$OUT/session.log"
  python -m harness.session "$st" --model "$MODEL" --out "$OUT" --tp "$TP" ${BAKED:+--baked "$BAKED"} 2>&1 | tee -a "$OUT/session.log"
done
