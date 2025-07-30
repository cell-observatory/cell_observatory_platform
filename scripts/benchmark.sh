#!/bin/bash

# USAGE: bash /clusterfs/nvme/hph/git_managed/cell_observatory_platform/scripts/benchmark.sh

# CFG="benchmarks/abc/benchmark_training_4d"
CFG="benchmarks/abc/benchmark_zarr_io.yaml"
# CFG="benchmarks/abc/benchmark_zarr_io_sweep.yaml"

# python3 /clusterfs/nvme/hph/git_managed/cell_observatory_platform/manager.py --config-name=${CFG}

# python3 /clusterfs/nvme/hph/git_managed/cell_observatory_platform/scripts/benchmark_training.py 
python3 /clusterfs/nvme/hph/git_managed/cell_observatory_platform/manager.py --config-name=${CFG}