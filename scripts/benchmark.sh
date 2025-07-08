#!/bin/bash

# USAGE: bash /clusterfs/nvme/hph/git_managed/cell_observatory_platform/scripts/benchmark.sh

CFG="benchmarks/benchmark_dataloaders_4d"
# CFG="benchmarks/benchmark_training_4d"

python3 /clusterfs/nvme/hph/git_managed/cell_observatory_platform/manager.py --config-name=${CFG}