#!/bin/bash
# Run stage-0c leaves one at a time (local launcher), each with its own log; print the bench_table row
# after each. Usage: run_queue.sh <log_dir> <cfg_dir> <name> [<name> ...]   (names without .yaml)
# Waits for each run's exit line before starting the next; never overwrites a log (attempt suffix).
LOGD=$1; CFGD=$2; shift 2
cd "$(dirname "$0")/../.." || exit 1
set -a; source .env; set +a; export PATH=$HOME/micromamba/bin:$PATH
for N in "$@"; do
  R=1; while [ -f "$LOGD/${N}_r$R.log" ]; do R=$((R+1)); done
  L="$LOGD/${N}_r$R.log"
  echo "=== $(date +%H:%M:%S) launching $N (attempt $R) -> $L"
  python manager.py --config-name="$CFGD/$N.yaml" > "$L" 2>&1; rc=$?
  echo "[run exited rc=$rc]" >> "$L"
  echo "=== $(date +%H:%M:%S) $N exited rc=$rc; OOM lines: $(grep -a -c OutOfMemoryError "$L"); errors: $(grep -a -c 'Error Snippet' "$L")"
  apptainer instance stop mysql >/dev/null 2>&1 || true
  OUT=$(grep -a -o "Output directory for training job: .*" "$L" | head -1 | sed 's/.*: //')
  [ -n "$OUT" ] && python scripts/utils/bench_table.py "$OUT" 2>/dev/null | tail -1 | cut -c1-400
done
echo "=== queue done $(date +%H:%M:%S)"
