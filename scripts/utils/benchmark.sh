#!/bin/bash

# USAGE: bash /clusterfs/nvme/hph/git_managed/cell_observatory_platform/scripts/utils/benchmark.sh

# CFG="benchmarks/abc/benchmark_training_4d"
# CFG="benchmarks/abc/benchmark_zarr_io.yaml"
# CFG="benchmarks/abc/benchmark_zarr_io_sweep.yaml"
# CFG="benchmarks/abc/benchmark_scaling_4d_base.yaml"
# CFG="benchmarks/abc/benchmark_training_base.yaml"

# CFG="benchmarks/abc/benchmark_dataloaders.yaml"
CFG="benchmarks/abc/benchmark_tensorstore.yaml"

python3 /clusterfs/nvme/hph/git_managed/cell_observatory_platform/manager.py --config-name=${CFG}