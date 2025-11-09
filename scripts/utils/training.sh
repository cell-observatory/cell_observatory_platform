#!/bin/bash

# --- ABC

# USAGE: bash /clusterfs/nvme/hph/git_managed/cell_observatory_platform/scripts/utils/training.sh

# CFG="test_pretrain_4d_mae_local.yaml"
# CFG="pretrain_jepa_local.yaml"
# CFG="experiments/abc/pretrain_mae_test_dali_07_13_2025.yaml"
# CFG="experiments/abc/pretrain_mae_test_torch_07_14_2025.yaml"
# CFG="benchmarks/abc/benchmark_training_4d.yaml"
# CFG="experiments/abc/pretrain_mae_test_tune_07_18_2025.yaml"
# CFG="experiments/abc/pretrain_mae_improve_utilization_07_23_2025.yaml"
# CFG="benchmarks/abc/benchmark_training_4d_dataloader.yaml"
# CFG="benchmarks/abc/benchmark_scaling_4d_base.yaml"

# python3 /clusterfs/nvme/hph/git_managed/cell_observatory_platform/manager.py --config-name=${CFG}

# --- Janelia 

# LSF interactive example cmd:
# bsub -Is -q gpu_h100_parallel -J "debug_job" -n 96 -gpu "num=8:mode=shared" -o "/groups/betzig/betziglab/hph/cell_observatory_project/log.%J" /bin/bash

# micromamba activate
# export PYTHONPATH="/groups/betzig/home/hamiltonh/git_managed/cell_observatory_platform"

# USAGE: bash /groups/betzig/home/hamiltonh/git_managed/cell_observatory_platform/scripts/utils/training.sh

# CFG="benchmarks/janelia/benchmark_training_dataloader.yaml"
# CFG="benchmarks/janelia/exp_10_22_2025_mae_3d_batch_size.yaml"
# CFG="experiments/janelia/exp_10_22_2025_hparam_sweep_mae/input_size_128x128x128_X_lr_X_masking_sweep.yaml"

# Janelia
# python3 /groups/betzig/home/hamiltonh/git_managed/cell_observatory_platform/manager.py --config-name=${CFG}

# --- CoreWeave 

# CFG="experiments/coreweave/tests/exp_11_05_2025_mae_3d_pretrain.yaml"
# CFG="experiments/coreweave/tests/exp_11_05_2025_mae_3d_pretrain_test_sweep.yaml"
# CFG="experiments/coreweave/tests/exp_11_08_25_test_interactive.yaml"

CFG="experiments\coreweave\exp_11_08_25_mae_lr_X_masking_ratio\exp_11_08_25_0p05_X_0p7.yaml"
# CFG="experiments\coreweave\exp_11_08_25_mae_lr_X_masking_ratio\exp_11_08_25_0p001_X_0p7.yaml"

# --- Linux

# USAGE: bash /work/cell_observatory_platform/scripts/utils/training.sh

# python3 /work/cell_observatory_platform/manager.py --config-name=${CFG}

# --- Windows

# USAGE: & "$Env:ProgramFiles\Git\bin\bash.exe" -lc '"/c/Users/HugoPatricHamilton/git_managed/cell-observatory/cell_observatory_platform/scripts/utils/training.sh"'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

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

# ALL JOBS: ai job list --proj cell-observatory
# LOGS: ai job follow --name <job_name> --project cell-observatory