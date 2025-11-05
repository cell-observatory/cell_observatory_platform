# !/bin/bash

# NOTE: when image is ready add the following line: --image cell-observatory
# for now run: uv pip install hydra dotenv omegaconf

# afterards do: uv pip install 'omegaconf==2.3.0' 'hydra-core==1.3.2' 'python-dotenv'

ai dev submit --project cell-observatory --ide-type code --name debug --gpus 0 --cpus-limit 8 --data project-cell-observatory-pvc-120t=/workspace/CellObservatoryData --data project-cell-observatory-pvc-120t=/workspace/cell_observatory_project --repo https://github.com/czi-ai/cell-observatory 