#!/bin/bash

uv venv

source .venv/bin/activate

uv pip install --upgrade \
  'astropy==7.1' 'connectorx==0.4.4' 'deepspeed==0.17.6' 'fvcore==0.1.5.post20221221' \
  'hydra-core==1.3.2' 'ipdb==0.13.13' 'line-profiler==5.0.0' 'line-profiler-pycharm==1.2.0' \
  'nvitop==1.5.3' 'omegaconf==2.3.0' 'py-libnuma==1.2' \
  'pytest==8.4.2' 'pytest-order==1.3.0' 'pytest-testmon==2.1.3' 'pytorchvideo==0.1.5' \
  'ray[all]==2.49.2' 'scikit-image==0.25.2' 'seaborn==0.13.2' 'supabase==2.19.0' \
  'tensorstore==0.1.77' 'tifffile==2025.9.20' 'timm==1.0.20' 'torchinfo==1.8.0' \
  'torchmetrics==1.8.2' 'trino==0.336.0' 'ujson==5.11.0' 'wandb==0.22.0' 'zarr==3.1.3' 'python-dotenv' 'polars==1.28.1' \
  'nvidia-dali-cuda130' 'tensorboard==2.16.2'