#!/bin/bash

# USAGE: bash /clusterfs/nvme/hph/git_managed/cell_observatory_platform/scripts/utils/training.sh

# CFG="test_pretrain_4d_mae_local.yaml"
# CFG="pretrain_jepa_local.yaml"
# CFG="experiments/abc/pretrain_mae_test_dali_07_13_2025.yaml"
# CFG="experiments/abc/pretrain_mae_test_torch_07_14_2025.yaml"
# CFG="benchmarks/abc/benchmark_training_4d.yaml"
# CFG="experiments/abc/pretrain_mae_test_tune_07_18_2025.yaml"
# CFG="experiments/abc/pretrain_mae_improve_utilization_07_23_2025.yaml"
# CFG="benchmarks/abc/benchmark_training_4d_dataloader.yaml"
CFG="benchmarks/abc/benchmark_scaling_4d_base.yaml"

python3 /clusterfs/nvme/hph/git_managed/cell_observatory_platform/manager.py --config-name=${CFG}