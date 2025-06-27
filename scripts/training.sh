#!/bin/bash

# USAGE: bash /clusterfs/nvme/hph/git_managed/cell_observatory_platform/scripts/training.sh

CFG="pretrain_mae_local.yaml"
# CFG="pretrain_jepa_local.yaml"

python3 /clusterfs/nvme/hph/git_managed/cell_observatory_platform/manager.py --config-name=${CFG}