# Compute estimates  

Accurate estimation of compute cost is key to plan experiments, allocate resources, and assess the feasibility of our approach.
Below we describe our framework for estimating the computational resources required for pretraining and finetuning vision models, starting from existing 2D ViT and scaling them to 3D, 4D, and 5D spatiotemporal biological data.

## Table of Contents

- [Compute estimates](#compute-estimates)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Simple assumptions to create a framework for estimating compute cost](#simple-assumptions-to-create-a-framework-for-estimating-compute-cost)
    - [Supported models](#supported-models)
  - [Quickstart](#quickstart)
  - [Usage](#usage)
  - [Examples](#examples)
    - [Analysis of published models](#analysis-of-published-models)
      - [WSP models](#wsp-models)
      - [MAE models](#mae-models)
        - [SSL stage](#ssl-stage)
        - [FT stage](#ft-stage)
    - [Analysis for custom models](#analysis-for-custom-models)
      - [ViT scaling on custom data (3D+time+channels) with custom patches](#vit-scaling-on-custom-data-3dtimechannels-with-custom-patches)
      - [MAE self-supervised pretraining with 75% masking](#mae-self-supervised-pretraining-with-75-masking)
      - [MAE finetuning analysis on bigger crops](#mae-finetuning-analysis-on-bigger-crops)

## Overview

### Simple assumptions to create a framework for estimating compute cost

- Scaling ViT to 3D and 4D data is done by extending the context window to encode the additional dimensions. See `profile.py` for details
- Several data types are supported to estimate model size and memory footprint ranging from `FP8` to `FP64`. `FP16` is used by default.
- Time estimates rely on `MFU` heuristics and GPU therotical peak `FLOPS`; default GPU used is `H100` with `989.4 TFLOPS` ([H100 whitepaper](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c))
- Training `FLOPs` per volume $\approx$3$\times$ inference FLOPs unless otherwise noted (e.g., MAE-FT freezes encoders).
- Parallelization across multiple GPUs is never perfect in practice, so the actual speedup might be less than linear with the number of GPUs.
- CSV columns include per-volume, per-batch, and dataset-scaled projections; see column names in `forecast.py` for details.

### Supported models

While we do not know the extact hyperparameters that would work best for 4D/5D spatiotemporal biological data, we can use current SOTA models to estimate a compute budget.

|  ViT  | Layers | Heads | Embedding Dim | MLP Dim |
|-------|--------|-------|---------------|---------|
| S     | 12     | 6     | 384           | 1536    |
| B     | 12     | 12    | 768           | 3072    |
| L     | 24     | 16    | 1024          | 4096    |
| H     | 32     | 16    | 1280          | 5120    |
| g     | 40     | 16    | 1408          | 6144    |
| G     | 48     | 16    | 1664          | 8192    |
| e     | 56     | 16    | 1792          | 15360   |
| 22B   | 48     | 48    | 6144          | 24576   |

[ViT-S/B/L/H (Dosovitskiy et al., 2021)](https://arxiv.org/abs/2010.11929)  
[ViT-g/G (Zhai et al., 2022)](https://arxiv.org/abs/2106.04560)  
[ViT-2/6B (Singh et al., 2023)](https://openaccess.thecvf.com//content/ICCV2023/papers/Singh_The_Effectiveness_of_MAE_Pre-Pretraining_for_Billion-Scale_Pretraining_ICCV_2023_paper.pdf)  
[ViT-e (Chen et al., 2022)](https://arxiv.org/abs/2209.06794)  
[ViT-22B (Dehghani et al., 2023)](https://arxiv.org/abs/2302.05442)

## Quickstart

```bash
python -m forecast -h
```

> [!IMPORTANT]
> You need to navigate to the `scaling` directory to run `forecast.py`

- `profile.py`: Core utilties to estimate FLOPs, parameters, and memory footprints, etc.
- `forecast.py`: Main CLI to create CSV summaries and plots, estimating compute requirements for ViT, MAE (SSL/FT), and Transformer-based encoder/decoders.
- `vis.py`: Plotting functions to render figures (PDF/PNG/SVG).

> [!NOTE]
> CLI will create CSVs and figures under `scaling/data/published_models/...`

`forecast.py` generates CSVs and figures for the chosen architecture(s).
- `--arch`: one of `vit`, `mae_ssl`, `mae_ft`, `transformer`, `published_models`
- `--ishape`: input dims as `t-z-y-x-c` (default: `16-128-128-128-2`)
- `--ipatch`: patch dims as `t-z-y-x-c` (default: `2-16-16-16-2`)
- `--dtype`: `float8|float16|float32|float64` (default: `float16`)
- `--batch_size`: used for certain derived metrics (default: `4096`)
- `--cost_h100_per_hr`: cost used in cost overlays (default: `49.24/8` [coreweave-cloud-compute](https://www.coreweave.com/pricing))
- `--outdir`: root output directory (default: `../scaling/data`)
- `--mask_ratio`: for MAE variants (default: `0.75` unless MAE function overrides)
- `--rgb`: toggle if inputs are Multichannel/RGB

## Usage

```python
from scaling import profile

# 1) Calcuate the number of tokens for a given image and patch size
volume_size = [1, 1, 224, 224, 3]   # [t, z, y, x, c]
patch_size  = [1, 1, 16, 16, 3]     # [t, z, y, x, c]
num_tokens = profile.patchify(volume_size, patch_size)
print("Tokens:", num_tokens)

# 2) Estimate total number of FLOPs for a given transformer-based encoder
total_flops = profile.encoder_transformer_flops(
    num_tokens=num_tokens, layers=12, embed_dim=768, heads=12, mlp_dim=3072
)
params = profile.encoder_transformer_params(layers=12, embed_dim=768, mlp_dim=3072)
print("Total FLOPs:", f"{total_flops:,}", "Parameters:", f"{params:,}")

# 3) Estimate memory footprint
inference_gb = profile.transformer_inference_memory_footprint(params, dtype="float16")
training_gb  = profile.transformer_training_memory_footprint(params, dtype="float16")
data_gb      = profile.data_memory_footprint(volume_size=[1, 1, 224, 224, 3], batch_size=4096, dtype="float16")
print(f"Inference GB: {inference_gb:.3f}, Training GB: {training_gb:.3f}, Data GB (per batch): {data_gb:.3f}")

# 4) Time estimates on a target GPU using a given MFU
seconds = profile.compute_time(flops=total_flops, gpu="H100", unit="seconds", mfu=0.6)
print(f"Per-volume seconds @ H100, MFU=0.6: {seconds:.3f}s")
```

## Examples

### Analysis of published models

#### WSP models

| Arch | t | z | y | x | c | Dataset | Dataset Size | Epochs | Steps | Batch Size |
|------|---|---|---|---|---|---------|-------------|--------|-------|------------|
| S    | 1 | 1 | 224 | 224 | 3 | ImageNet-21K  | 14,197,122    | 7    | 24,273     | 4096      |
| B    | 1 | 1 | 224 | 224 | 3 | ImageNet-21K  | 14,197,122    | 7    | 24,273     | 4096      |
| L    | 1 | 1 | 224 | 224 | 3 | JFT-300M      | 303,000,000   | 14   | 1,035,156  | 4096      |
| H    | 1 | 1 | 224 | 224 | 3 | JFT-300M      | 303,000,000   | 14   | 1,035,156  | 4096      |
| g    | 1 | 1 | 224 | 224 | 3 | JFT-1B        | 3,000,000,000 | 5.46 | 4,000,000  | 4096      |
| G    | 1 | 1 | 224 | 224 | 3 | JFT-3B        | 3,000,000,000 | 6.82 | 5,000,000  | 4096      |
| e    | 1 | 1 | 224 | 224 | 3 | JFT-3B        | 3,000,000,000 | 5.46 | 1,000,000  | 16384     |
| 22B  | 1 | 1 | 224 | 224 | 3 | JFT-4B        | 4,000,000,000 | 2.88 | 177,000    | 65000     |

#### MAE models

##### SSL stage

| Arch | t | z | y | x | c | Dataset | Dataset Size | Epochs | Steps | Batch Size | Mask Ratio |
|------|---|---|---|---|---|---------|-------------|--------|-------|------------|------------|
| S    | 1 | 1 | 224 | 224 | 3 | ImageNet-1K   | 1,281,167     | 1600  | 500,457   | 4096     | 0.75 |
| B    | 1 | 1 | 224 | 224 | 3 | ImageNet-1K   | 1,281,167     | 1600  | 500,457   | 4096     | 0.75 |
| L    | 1 | 1 | 224 | 224 | 3 | ImageNet-1K   | 1,281,167     | 1600  | 500,457   | 4096     | 0.75 |
| H    | 1 | 1 | 224 | 224 | 3 | Instagram-3B  | 3,000,000,000 | 1     | 732,423   | 4096     | 0.75 |
| 2B   | 1 | 1 | 224 | 224 | 3 | Instagram-3B  | 3,000,000,000 | 4     | 2,929,692 | 4096     | 0.75 |
| 6B   | 1 | 1 | 224 | 224 | 3 | Instagram-3B  | 3,000,000,000 | 4     | 2,929,692 | 4096     | 0.75 |

##### FT stage

| Arch | t | z | y | x | c | Dataset | Dataset Size | Epochs | Steps | Batch Size |
|------|---|---|---|---|---|---------|-------------|--------|-------|------------|
| S    | 1 | 1 | 518 | 518 | 3 | ImageNet-1K   | 1,281,167     | 50   | 62,597    | 1024      |
| B    | 1 | 1 | 518 | 518 | 3 | ImageNet-1K   | 1,281,167     | 50   | 62,597    | 1024      |
| L    | 1 | 1 | 518 | 518 | 3 | ImageNet-1K   | 1,281,167     | 50   | 62,597    | 1024      |
| H    | 1 | 1 | 518 | 518 | 3 | ImageNet-1K   | 1,281,167     | 50   | 62,597    | 1024      |
| 2B   | 1 | 1 | 518 | 518 | 3 | ImageNet-1K   | 1,281,167     | 50   | 62,597    | 1024      |
| 6B   | 1 | 1 | 518 | 518 | 3 | ImageNet-1K   | 1,281,167     | 50   | 62,597    | 1024      |

> !NOTE: results will be saved in the following format:

```shell
scaling/data/published_models/{ViT|MAE-SSL|MAE-FT}/
  ├── published_models_{mfu}_mfu.csv
  └── figures for dataset size, volumes, time, TFLOPs, cost
```

### Analysis for custom models

#### ViT scaling on custom data (3D+time+channels) with custom patches

```bash
python -m forecast \
  --arch vit \
  --ishape 16-128-128-128-2 \
  --ipatch 4-16-16-16-2 \
  --dtype float16 \
  --outdir ./data/ \
  --rgb
```

#### MAE self-supervised pretraining with 75% masking

```bash
python -m forecast \
  --arch mae_ssl \
  --ishape 16-128-128-128-2 \
  --ipatch 4-16-16-16-2 \
  --mask_ratio 0.75 \
  --outdir ./data/ \
  --rgb
```

#### MAE finetuning analysis on bigger crops

```bash
python -m forecast \
  --arch mae_ft \
  --ishape 16-128-512-512-2 \
  --ipatch 4-116-16-16-2 \
  --outdir ./data/ \
  --rgb
```

> !NOTE: results will be saved in the following format:

```shell
scaling/data/{arch}/size-{t}-{z}-{y}-{x}-{c}/patch-{pt}-{pz}-{py}-{px}-{pc}/
  ├── summary/
  │   └── {encoder|decoder|autoencoder}/
  │       ├── training_gpu_days_per_epoch_{dataset_size}_{default|dark_background}.{pdf,png,svg}
  └── {arch}_{mfu}_mfu.csv  # aggregated metrics for mfu∈{0.3,0.6,0.9}
```
