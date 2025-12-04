#!/bin/bash

# --- Janelia

# USAGE: bash /groups/betzig/home/hamiltonh/git_managed/cell_observatory_platform/scripts/utils/inference.sh

# export PYTHONPATH="/groups/betzig/home/hamiltonh/git_managed:/groups/betzig/home/hamiltonh/git_managed/cell_observatory_platform"

# CFG="experiments/janelia/exp_10_28_2025_test_pipeline/test_inference_pretrain.yaml"

# python3 /groups/betzig/home/hamiltonh/git_managed/cell_observatory_platform/manager.py --config-name=${CFG}

# USAGE: bash /work/cell_observatory_platform/scripts/utils/inference.sh

export PYTHONPATH="/work:/work/cell_observatory_platform"

CFG="experiments/coreweave/tests/exp_11_13_25_test_inference.yaml"

python3 /work/cell_observatory_platform/manager.py --config-name=${CFG}