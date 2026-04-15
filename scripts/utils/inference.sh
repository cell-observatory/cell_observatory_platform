#!/bin/bash

# --- Janelia

# USAGE: bash /groups/betzig/home/hamiltonh/git_managed/cell_observatory_platform/scripts/utils/inference.sh

# export PYTHONPATH="/groups/betzig/home/hamiltonh/git_managed:/groups/betzig/home/hamiltonh/git_managed/cell_observatory_platform"

# CFG="experiments/janelia/exp_10_28_2025_test_pipeline/test_inference_pretrain.yaml"

# python3 /groups/betzig/home/hamiltonh/git_managed/cell_observatory_platform/manager.py --config-name=${CFG}

# USAGE: bash /work/cell_observatory_platform/scripts/utils/inference.sh

# CFG="experiments/abc/tests/test_instance_segmentation_maskdino_inference_fish1_roi1.yaml"
CFG="experiments/abc/tests/test_semantic_segmentation_mask2former_inference.yaml"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MANAGER_PY="$REPO_ROOT/cell_observatory_platform/manager.py"
if command -v cygpath >/dev/null 2>&1; then
  MANAGER_PY="$(cygpath -u "$MANAGER_PY")"
fi

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/cell_observatory_platform"


if command -v uv >/dev/null 2>&1; then
  exec uv run python "$MANAGER_PY" --config-name="$CFG"
elif command -v python3 >/dev/null 2>&1; then
  exec python3 "$MANAGER_PY" --config-name="$CFG"
else
  exec python "$MANAGER_PY" --config-name="$CFG"
fi