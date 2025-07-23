#!/bin/bash

# USAGE: bash /clusterfs/nvme/hph/git_managed/cell_observatory_platform/scripts/plot_benchmark.sh

# CFG="benchmarks/benchmark_dataloaders_4d"
# CFG="benchmarks/benchmark_training_4d"
CFG="plots/benchmark_training_4d"

python3 /clusterfs/nvme/hph/git_managed/cell_observatory_platform/scripts/plot_benchmark.py --config-name=${CFG}