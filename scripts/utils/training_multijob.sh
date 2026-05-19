set -euo pipefail

# Run every *.yaml in a folder by calling manager.py --config-name="<hydra-config-name>"
#
# USAGE (Windows Git Bash):
#   & "$Env:ProgramFiles\Git\bin\bash.exe" -lc 'CFG_DIR="/c/Users/.../my_folder/instance_seg" "/c/Users/.../scripts/utils/training_multijob.sh"'
#
# Optional:
#   DRY_RUN=1  (print commands without running)
#   FAIL_FAST=1 (default) stops on first failure; set FAIL_FAST=0 to continue

# ---- ---- ---- COREWEAVE ---- ---- ----

# CFG_DIR="experiments/coreweave/exp_12_19_25_mae_lr_X_finetune_task"

# ---- ---- ---- ---- ---- ---- ---- ----

# ---- ---- ---- JANELIA ---- ---- ----
: "${CFG_DIR:=experiments/janelia/profiling/allreduce_spike_experiment}"

# ---- ---- ---- ---- ---- ---- ---- ----

: "${DRY_RUN:=0}"
: "${FAIL_FAST:=1}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

MANAGER_PY="$REPO_ROOT/manager.py"
CONFIGS_ROOT="$REPO_ROOT/configs"

# Convert to unix path if cygpath exists (Git Bash on Windows)
if command -v cygpath >/dev/null 2>&1; then
  MANAGER_PY="$(cygpath -u "$MANAGER_PY")"
  REPO_ROOT_U="$(cygpath -u "$REPO_ROOT")"
  CONFIGS_ROOT_U="$(cygpath -u "$CONFIGS_ROOT")"
  CFG_DIR_U="$(cygpath -u "$CFG_DIR")"
else
  REPO_ROOT_U="$REPO_ROOT"
  CONFIGS_ROOT_U="$CONFIGS_ROOT"
  CFG_DIR_U="$CFG_DIR"
fi

# Resolve config dir against configs root so find works regardless of cwd
CFG_FULL="$CONFIGS_ROOT_U/$CFG_DIR_U"

echo "[training_multijob.sh] Repo root:    $REPO_ROOT_U"
echo "[training_multijob.sh] Manager:      $MANAGER_PY"
echo "[training_multijob.sh] Configs root: $CONFIGS_ROOT_U"
echo "[training_multijob.sh] Config dir:   $CFG_FULL"

# Pick runner
run_python() {
  local cfg="$1"
  if command -v uv >/dev/null 2>&1; then
    uv run python "$MANAGER_PY" --config-name="$cfg"
  elif command -v python3 >/dev/null 2>&1; then
    python3 "$MANAGER_PY" --config-name="$cfg"
  else
    python "$MANAGER_PY" --config-name="$cfg"
  fi
}

# Collect YAMLs from the given directory (path from configs root, not cwd)
shopt -s nullglob
mapfile -t yamls < <(find "$CFG_FULL" -maxdepth 1 -type f \( -name "*.yaml" -o -name "*.yml" \) ! -name "base_*.yaml" | sort)

if [[ ${#yamls[@]} -eq 0 ]]; then
  echo "[training_multijob.sh] No YAML files found in: $CFG_FULL"
  exit 1
fi

echo "[training_multijob.sh] Found ${#yamls[@]} YAML files."

failures=0
for f in "${yamls[@]}"; do
  # Compute Hydra config-name:
  # Prefer path relative to configs/ if file is under it; otherwise use absolute path.
  cfg_name="$f"
  if [[ "$f" == "$CONFIGS_ROOT_U/"* ]]; then
    cfg_name="${f#"$CONFIGS_ROOT_U/"}"
  fi

  echo "------------------------------------------------------------"
  echo "[training_multijob.sh] Running: $cfg_name"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] would run: manager.py --config-name=\"$cfg_name\""
    continue
  fi

  if run_python "$cfg_name"; then
    echo "[training_multijob.sh] OK: $cfg_name"
  else
    echo "[training_multijob.sh] FAIL: $cfg_name"
    failures=$((failures + 1))
    if [[ "$FAIL_FAST" == "1" ]]; then
      exit 2
    fi
  fi
done

echo "------------------------------------------------------------"
if [[ "$failures" -gt 0 ]]; then
  echo "[training_multijob.sh] Done with failures: $failures"
  exit 2
fi

echo "[training_multijob.sh] Done. All configs ran successfully."