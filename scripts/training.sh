#!/bin/bash

# USAGE: bash /clusterfs/nvme/hph/git_managed/cell_observatory_platform/training/training.sh

# CFG="pretrain_mae.yaml"
CFG="pretrain_jepa.yaml"

python3 /clusterfs/nvme/hph/git_managed/cell_observatory_platform/cluster/manager.py --config-name=${CFG}