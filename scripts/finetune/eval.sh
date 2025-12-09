#!/bin/bash

# --- ABC

# USAGE: bash /clusterfs/nvme/hph/git_managed/cell_observatory_platform/scripts/eval.sh

export PYTHONPATH="/clusterfs/nvme/hph/git_managed:/clusterfs/nvme/hph/git_managed/cell_observatory_platform/cell_observatory_platform"

CFG="experiments/abc/evals/test_evals.yaml"

python3 /clusterfs/nvme/hph/git_managed/cell_observatory_platform/manager.py --config-name=${CFG}