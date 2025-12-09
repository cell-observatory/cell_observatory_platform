#!/bin/bash

# --- ABC

# USAGE: bash /clusterfs/nvme/hph/git_managed/cell_observatory_platform/scripts/train.sh

# export PYTHONPATH="/clusterfs/nvme/hph/git_managed:/clusterfs/nvme/hph/git_managed/cell_observatory_platform/cell_observatory_platform"

# --- 4D ----

# CFG="experiments/abc/4D/mae/channel_split/test_channel_split_linear_10_13_25.yaml"
# CFG="experiments/abc/4D/mae/channel_split/test_channel_split_vit_10_13_25.yaml"
# CFG="experiments/abc/4D/mae/channel_split/test_channel_split_dpt_10_13_25.yaml"

# CFG="experiments/abc/4D/mae/upsample_space/test_linear_10_13_25.yaml"
# CFG="experiments/abc/4D/mae/upsample_space/test_vit_10_13_25.yaml"

# CFG="experiments/abc/4D/mae/upsample_time/test_vit_10_13_25.yaml"

# CFG="experiments/abc/4D/mae/upsample_spacetime/test_vit_10_13_25.yaml"

# CFG="experiments/abc/4D/jepa/channel_split/test_linear_10_13_25.yaml"
# CFG="experiments/abc/4D/jepa/channel_split/test_vit_10_13_25.yaml"

# CFG="experiments/abc/4D/jepa/upsample_space/test_linear_10_03_25.yaml"
# CFG="experiments/abc/4D/jepa/upsample_space/test_vit_10_03_25.yaml"

# CFG="experiments/abc/4D/jepa/upsample_time/test_vit_10_03_25.yaml"

# CFG="experiments/abc/4D/jepa/upsample_spacetime/test_vit_10_03_25.yaml"

# --- 3D ----

# CFG="experiments/abc/3D/mae/channel_split/test_channel_split_linear_10_13_25.yaml"
# CFG="experiments/abc/3D/mae/channel_split/test_channel_split_vit_10_13_25.yaml"

# CFG="experiments/abc/3D/mae/upsample_space/test_linear_10_13_25.yaml"
# CFG="experiments/abc/3D/mae/upsample_space/test_vit_10_13_25.yaml"

# CFG="experiments/abc/3D/jepa/channel_split/test_linear_10_13_25.yaml"
# CFG="experiments/abc/3D/jepa/channel_split/test_vit_10_13_25.yaml"

# CFG="experiments/abc/3D/jepa/upsample_space/test_linear_10_03_25.yaml"
# CFG="experiments/abc/3D/jepa/upsample_space/test_vit_10_03_25.yaml"

# python3 /clusterfs/nvme/hph/git_managed/cell_observatory_platform/manager.py --config-name=${CFG}

# --- CoreWeave ----

# CFG="experiments/coreweave/3D/mae/channel_split/test_channel_split_vit_10_13_25.yaml"

# CFG="experiments/coreweave/3D/mae/channel_split/exp_11_12_25_lr_sweep/baseline_0p01.yaml"
# CFG="experiments/coreweave/3D/mae/channel_split/exp_11_12_25_lr_sweep/baseline_0p001.yaml"

# CFG="experiments/coreweave/3D/mae/channel_split/exp_11_12_25_lr_sweep/pretrain_lr_0p001_mask_0p8_LR_0p01.yaml"
CFG="experiments/coreweave/3D/mae/channel_split/exp_11_12_25_lr_sweep/pretrain_lr_0p001_mask_0p8_LR_0p001.yaml"

# CFG="experiments/coreweave/tests/exp_11_13_25_test_inference.yaml"

# --- Linux

# USAGE: bash /work/cell_observatory_platform/scripts/train.sh

# export PYTHONPATH="/work:/work/cell_observatory_platform"

# python3 /work/cell_observatory_platform/manager.py --config-name=${CFG}

# --- Windows

# USAGE: & "$Env:ProgramFiles\Git\bin\bash.exe" -lc '"/c/Users/HugoPatricHamilton/git_managed/cell-observatory/cell_observatory_platform/scripts/train.sh"'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

MANAGER_PY="$REPO_ROOT/cell_observatory_platform/manager.py"
if command -v cygpath >/dev/null 2>&1; then
  MANAGER_PY="$(cygpath -u "$MANAGER_PY")"
fi

echo "[training.sh] Repo root: $REPO_ROOT"
echo "[training.sh] Manager:   $MANAGER_PY"
echo "[training.sh] Config:    $CFG"

if command -v uv >/dev/null 2>&1; then
  exec uv run python "$MANAGER_PY" --config-name="$CFG"
elif command -v python3 >/dev/null 2>&1; then
  exec python3 "$MANAGER_PY" --config-name="$CFG"
else
  exec python "$MANAGER_PY" --config-name="$CFG"
fi