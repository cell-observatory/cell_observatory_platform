#!/bin/bash

# NOTE: when image is ready add the following line:
#    --image cell-observatory
# for now run:
#   uv pip install hydra dotenv omegaconf

ai dev submit \
    --project cell-observatory \
    --name debug_job \
    --gpus 0 \
    --cpus-limit 8 \
    --data project-cell-observatory-pvc-120t=/workspace/CellObservatoryData \
    --data project-cell-observatory-pvc-120t=/workspace/cell_observatory_project \
    --repo https://github.com/czi-ai/cell-observatory