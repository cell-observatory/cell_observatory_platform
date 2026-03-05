
# Cell Observatory Platform
**The Cell Observatory Platform** is a comprehensive framework for training and evaluating machine learning models on biological image and video datasets. Built with [PyTorch](https://pytorch.org/), orchestrated and scaled with [Ray](https://www.ray.io/), model sharding via [DeepSpeed](https://www.deepspeed.ai/) or native PyTorch parallelism ([TorchTitan](https://github.com/pytorch/torchtitan)), and flexibly configured using [Hydra](https://hydra.cc/), it provides a modular architecture for easy customization and extension.

- [Cell Observatory Platform](#cell-observatory-platform)
- [Installation](#installation)
  - [Docker images](#docker-images)
  - [Clone the repository to your host system](#clone-the-repository-to-your-host-system)
  - [Setup Supabase and W\&B accounts](#setup-supabase-and-wb-accounts)
  - [Setup environment variables](#setup-environment-variables)
- [Running docker image](#running-docker-image)
  - [Running docker image on a cluster via apptainer](#running-docker-image-on-a-cluster-via-apptainer)
    - [amd64/x86\_64](#amd64x86_64)
    - [arm64/aarch64](#arm64aarch64)
  - [Building a new apptainer image with a different torch version](#building-a-new-apptainer-image-with-a-different-torch-version)
- [Get started](#get-started)
  - [Local setup](#local-setup)
  - [Cluster setup](#cluster-setup)
- [Architecture Overview](#architecture-overview)
- [Data Pipeline](#data-pipeline)
  - [Databases](#databases)
  - [Ray Dataloader](#ray-dataloader)
  - [Preprocessors](#preprocessors)
  - [Transforms](#transforms)
  - [MaskGenerator](#maskgenerator)
- [Models](#models)
  - [Pretraining Models](#pretraining-models)
  - [Detection \& Segmentation](#detection--segmentation)
  - [Backbones](#backbones)
  - [Layers](#layers)
- [Training](#training)
  - [EpochBasedTrainer (DeepSpeed)](#epochbasedtrainer-deepspeed)
  - [ParallelEpochBasedTrainer (TorchTitan)](#parallelepochbasedtrainer-torchtitan)
  - [Hooks](#hooks)
- [Logging \& Experiment Tracking](#logging--experiment-tracking)
  - [Event Recording](#event-recording)
  - [W\&B Integration](#wb-integration)
  - [Metrics Processing](#metrics-processing)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Profiling](#profiling)
- [Configuration layout](#configuration-layout)
- [License](#license)


# Installation

## Docker [images](https://github.com/cell-observatory/cell_observatory_platform/pkgs/container/cell_observatory_platform)
Our prebuilt image with Python, Torch, and all packages installed for you.
```shell
docker pull ghcr.io/cell-observatory/cell_observatory_platform:develop_torch_26_01
```

## Clone the repository to your host system
```shell
git clone --recurse-submodules https://github.com/cell-observatory/cell_observatory_platform.git
```

To later update to the latest, greatest.
```shell
git pull --recurse-submodules
```

> [!NOTE]
> If you want to run a local version of the image, see the [Dockerfile](https://github.com/cell-observatory/cell_observatory_platform/blob/main/Dockerfile)

## Setup Supabase and W&B accounts

You will need to create a Supabase and W&B account to use the platform.
Supabase can be found at [Cell Observatory Database](https://supabase.com/dashboard/org/yrgvnbckfmhfyxgzzkqb), and W&B can be found at [Cell Observatory Dashboard](https://wandb.ai/cell-observatory).

Once you have created your Supabase and W&B accounts, you'll need to add your API keys in the environment variables as described below.

## Setup environment variables
Rename `.env.example` file to `.env` which will be automatically loaded into the container and will be gitignored. The Supabase related environment variables enable database functionality. The W&B API key enables logging functionality. The `REPO_NAME`, `DATA_DIR`, and `STORAGE_SERVER_DIR` environment variables are leverged in the `configs/paths` configuration files to ensure that jobs run and save outputs as expected.

> [!NOTE]
> `STORAGE_SERVER_DIR` is usually set to the root directory of your files. See example below:
> ```shell
> STORAGE_SERVER_DIR="/clusterfs/scratch/user/"
> REPO_DIR="/clusterfs/scratch/user/cell_observatory_platform"
> DATA_DIR="/clusterfs/scratch/user/cell_observatory_data"
> ```

```shell
SUPABASE_USER=REPLACE_ME_WITH_YOUR_SUPABASE_USERNAME
SUPABASE_PASS=REPLACE_ME_WITH_YOUR_SUPABASE_PASSWORD
TRINO_USER=REPLACE_ME_WITH_YOUR_TRINO_USERNAME
TRINO_PASS=REPLACE_ME_WITH_YOUR_TRINO_PASSWORD
SUPABASE_STAGING_ID=REPLACE_ME_WITH_YOUR_SUPABASE_STAGING_ID
SUPABASE_PROD_ID=REPLACE_ME_WITH_YOUR_SUPABASE_PROD_ID
WANDB_API_KEY=REPLACE_ME_WITH_YOUR_WANDB_API_KEY

SUPABASE_STAGING_URI="postgresql://${SUPABASE_USER}.${SUPABASE_STAGING_ID}:${SUPABASE_PASS}@aws-0-us-east-1.pooler.supabase.com:5432/postgres"
SUPABASE_PROD_URI="postgresql://${SUPABASE_USER}.${SUPABASE_PROD_ID}:${SUPABASE_PASS}@aws-0-us-east-1.pooler.supabase.com:5432/postgres"

REPO_NAME=cell_observatory_platform # TODO: replace with your repo name if you renamed it
REPO_DIR=REPLACE_ME_WITH_YOUR_ROOT_REPO_DIR
DATA_DIR=REPLACE_ME_WITH_YOUR_ROOT_DATA_DIR_WHERE_DATA_WILL_BE_SAVED
STORAGE_SERVER_DIR=REPLACE_ME_WITH_YOUR_STORAGE_SERVER_DIR_WHERE_DATA_SERVER_IS_MOUNTED
PYTHONPATH=REPLACE_ME_WITH_YOUR_ROOT_REPO_DIR
````

> [!IMPORTANT]
> Username/password and IDs for supabase will be provided upon request. 


# Running docker image

To run docker image, cd to repo directory or replace `$(pwd)` with your local path for the repository.
```shell
docker run --network host -u 1000 --privileged -v $(pwd):/workspace/cell_observatory_platform -w /workspace/cell_observatory_platform --env PYTHONUNBUFFERED=1 --pull missing -it --rm  --ipc host --gpus all ghcr.io/cell-observatory/cell_observatory_platform:develop_torch_26_01 bash
```

## Running docker image on a cluster via apptainer
Running an image on a cluster typically requires an apptainer version of the image, which can be generated by:

### amd64/x86_64
```shell
apptainer pull --arch amd64 --force develop_torch_26_01.sif docker://ghcr.io/cell-observatory/cell_observatory_platform:develop_torch_26_01
```

### arm64/aarch64
```shell
apptainer pull --arch arm64 --force develop_torch_26_01.sif docker://ghcr.io/cell-observatory/cell_observatory_platform:develop_torch_26_01
```

## Building a new apptainer image with a different torch version 
First, you need to build an apptainer image for torch from the containers provided by Nvidia (e.g., `26.01-py3` from this [catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags)):
``` shell
apptainer pull --arch amd64 --force pytorch_26.01-py3.sif docker://nvcr.io/nvidia/pytorch:26.01-py3
```

Then you can run the following command to build a complete image:  
``` shell
apptainer build --arch amd64 --nv --force develop_torch_26_01.sif apptainerfile.def
```

> [!IMPORTANT]
> Make sure to pass in the right argument for your system  (`amd64` or `arm64`)

# Get started

All jobs are orchestrated on top of a **Ray cluster** and launched through our `manager.py` script, which facilitates cluster resource allocation and Ray cluster setup. You may decide whether to run jobs locally or on a cluster by setting the `launcher_type` variable in `configs/clusters/*.yaml`. We show how to run jobs locally and on SLURM or LSF clusters below.

## Local setup

Example job configs are located in the `configs/experiments` folder. For local jobs, you can use our existing `configs/paths/local.yaml` and `configs/clusters/local.yaml` configurations.

### 1. Update experiment name
```yaml
experiment_name: test_cell_observatory_platform
wandb_project: test_cell_observatory_platform
```

### 2. Update your paths
```yaml
paths:
  outdir: ${paths.data_path}/pretrained_models/${experiment_name}
  resume_checkpointdir: null 
  pretrained_checkpointdir: null
```

### 3. Edit resource requirements
```yaml
clusters:
  batch_size: 2
  worker_nodes: 1
  gpus_per_worker: 1
  cpus_per_gpu: 4
  mem_per_cpu: 16000
```

### 4. Run local training job
```bash
python manager.py --config-name=configs/test_pretrain_4d_mae_local.yaml
```

### 5. Launch multiple training jobs or Ray Tune jobs
To launch multiple training jobs, set `run_type` to `multi_run` and define a `runs` list. For Ray Tune hyperparameter sweeps, set `run_type` to `tune`.

## Cluster setup

Running a job on a cluster is very similar to the local setup. Override the `defaults` in your config file to match your cluster:

### SLURM Setup
```yaml
defaults:
  - clusters: abc_a100
  - paths: abc
```

### LSF Setup
```yaml
defaults:
  - clusters: janelia_h100
  - paths: janelia
```

# Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Ray Cluster                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         manager.py                                      ││
│  │                    (SLURM / LSF / Local)                                ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    │                                        │
│           ┌────────────────────────┼────────────────────────┐               │
│           ▼                        ▼                        ▼               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │ EpochBased      │    │ ParallelEpoch   │    │   Inferencer    │          │
│  │ Trainer         │    │ BasedTrainer    │    │                 │          │
│  │ (DeepSpeed)     │    │ (TorchTitan)    │    │ (Distributed)   │          │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘          │
│           │                        │                        │               │
│           └────────────────────────┼────────────────────────┘               │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      Ray Data Pipeline                                  ││
│  │  LoaderActor → SharedMemory → CollatorActor → DeviceBuffer → GPU        ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

# Data Pipeline

The data pipeline is built on **Ray Data** with a custom queue-based system for high-throughput data loading:

```
Database (Supabase/Local CSV) -> LoaderActor -> Host SharedMemory Buffer -> CollatorActor -> Device Buffer -> Preprocessor -> Model
```

## Databases

**`ParentDatabase`** / **`SupabaseDatabase`**: Flexible database classes supporting both remote (Supabase/PostgreSQL) and local (cached CSV) data sources.

- **Remote mode**: Queries Supabase via ConnectorX (Arrow path) for high-performance data fetching
- **Local mode**: Loads from cached CSV + JSON config for offline/fast iteration
- **Data filtering**: ROI/tile/HPF selection, occupancy thresholds, CDF thresholds, annotation filtering

```python
# Example database config
datasets:
  databases:
    _target_: data.databases.supabase_database.SupabaseDatabase
    use_cached_hypercubes_dataframe: true
    hypercubes_dataframe_path: ${paths.data_path}/databases/hypercubes.csv
    server_folder_path: /groups/betzig/data
```

## Ray Dataloader

The Ray-based dataloader uses a multi-actor architecture for maximum throughput:

**`LoaderActor`**: Reads hypercubes from Zarr via TensorStore into pinned shared memory buffers.

**`CollatorActor`** / **`FinetuneCollatorActor`**: Transfers batches from host shared memory to GPU device buffers with optional transforms, mask extraction, and target building.

## Preprocessors

Preprocessors provide a unified interface for task-specific data preparation:

| Preprocessor | Task | Description |
|-------------|------|-------------|
| `RayPreprocessor` | Pretraining | dtype normalization, masking, transforms |
| `ChannelSplitPreprocessor` | Channel Split | Predicts per-channel from averaged input |
| `UpsamplePreprocessor` | Super-resolution | NA-mask downsampling for space/time upsampling |
| `InstanceSegmentationPreprocessor` | Instance Seg | Mask/bbox extraction, target building |

## Transforms

Transforms can be applied in either the **Collator** (CPU, during data loading) or the **Preprocessor** (GPU, before model forward). Available transforms in `data/transforms/`:

- **`Resize`**: Spatial resizing with mask/bbox scaling
- **`Crop`**: Random and center cropping
- **`Normalize`**: Percentile-based normalization
- **`ProbabilisticChoice`**: Randomly select between transform pipelines

## MaskGenerator

Generates patch-level masks for self-supervised learning with explicit time/space awareness:

- `BLOCKED` / `BLOCKED_TIME_ONLY` / `BLOCKED_SPACE_ONLY`: Block-based masking
- `RANDOM` / `RANDOM_SPACE_ONLY`: MAE-style masking
- `BLOCKED_PATTERNED`: Deterministic time downsampling patterns

# Models

## Pretraining Models

| Model | Location | Description |
|-------|----------|-------------|
| **MAE** | `models/meta_arch/maskedautoencoder.py` | Masked Autoencoder for 3D/4D volumes |
| **JEPA** | `models/meta_arch/jepa.py` | Joint-Embedding Predictive Architecture |

## Detection & Segmentation

| Model | Location | Description |
|-------|----------|-------------|
| **plainDETR** | `models/meta_arch/plainDETR.py` | 3D object detection  |
| **MaskDINO** | `models/meta_arch/maskdino.py` | 3D instance segmentation |
| **Mask2Former** |  | 3D semantic segmentation |

## Backbones

- **ViT** (`models/backbones/vit.py`): Vision Transformer with RoPE/sincos positional encoding
- **ConvNeXt** (`models/backbones/convnext.py`): ConvNeXt backbone
- **MaskedEncoder** (`models/backbones/maskedencoder.py`): Encoder with masking support

## Layers

Key layer implementations:
- **Attention**: Multi-head self-attention with flash attention support and deformable attention 
- **Transformer**: Standard and deformable transformer blocks
- **Patch Embeddings**: 3D/4D patch embedding with multiple layout support
- **Positional Encoding**: Sinusoidal, learned, and RoPE encodings
- **Matchers**: Hungarian matcher for detection/segmentation

# Training

## EpochBasedTrainer (DeepSpeed)

Standard training loop with DeepSpeed ZeRO optimization:



**Features:**
- DeepSpeed ZeRO stages 1-3
- Mixed precision (bf16/fp16)
- Checkpoint saving/resuming
- Hook-based extensibility

## ParallelEpochBasedTrainer (TorchTitan)

Advanced parallelism support via TorchTitan integration (**work in progress**):



**Features:**
- **TP** (Tensor Parallelism): Shards model tensor
- **CP** (Context Parallelism): Shards sequence dimension
- **FSDP** (Model and Data Parallelism): FSDP-based sharding
- **Torch Compile** support
- **Activation checkpointing**

## Hooks

The training loop is extensible via a priority-based hook system. Hooks can execute at various points: `before_train`, `before_epoch`, `before_step`, `after_backward`, `after_step`, `after_epoch`, `after_train`, and validation/test equivalents.

| Hook | Purpose |
|------|---------|
| `IterationTimer` | Tracks step/epoch/validation timing, logs ETA |
| `LRScheduler` | Executes LR scheduler steps, logs learning rate |
| `WeightDecayScheduleHook` | Updates weight decay on schedule |
| `PeriodicWriter` | Writes metrics to loggers at epoch end |
| `PeriodicCheckpointer` | Saves checkpoints at configurable intervals |
| `BestCheckpointer` | Reports best checkpoint to Ray for model selection |
| `BestMetricSaver` | Tracks and updates best validation metric |
| `TorchMemoryStats` | Logs CUDA memory usage (reserved/allocated) |
| `TorchProfiler` | PyTorch profiler with TensorBoard traces and memory snapshots |
| `NsysProfilerHook` | NVIDIA Nsight Systems profiling for GPU timeline analysis |
| `EarlyStopHook` | Stops training if validation metric plateaus |
| `EMASchedulerHook` | Updates EMA beta for target networks (JEPA) |
| `FreeDeviceBufferHook` | Releases Ray device buffers to prevent deadlocks |
| `AnomalyDetector` | Detects NaN/Inf losses with `torch.autograd.detect_anomaly` |
| `GarbageCollectionHook` | Periodic synchronized GC to prevent memory fragmentation |
| `MemoryDebugHook` | Detailed memory dumps (proc, CUDA, Ray, /dev/shm) |
| `AdjustTimeoutHook` | Adjusts distributed timeout for long-running ops |

# Logging & Experiment Tracking

## Event Recording

The `EventRecorder` is the central hub for collecting metrics during training. It supports step-scoped and epoch-scoped scalars with configurable reduction methods (mean, median, sum, min, max).

```python
# Record a scalar at step scope
trainer.event_recorder.put_scalar("loss", loss_value, scope="step")

# Record multiple scalars with a prefix
trainer.event_recorder.put_scalars(scope="epoch", prefix="val_", accuracy=0.95, f1=0.92)
```

## W&B Integration

The `WandBEventWriter` provides seamless Weights & Biases integration.



**Features:**
- Automatic login via `WANDB_API_KEY` from `.env`
- Custom metric namespacing (`step/*`, `epoch/*`)
- Tags and notes for experiment organization

The `LocalEventWriter` saves metrics to CSV files for offline analysis.



## Metrics Processing

For advanced training loops (TorchTitan), the `MetricsProcessor` computes detailed performance metrics (**work in progress**):

- **Throughput**: Tokens per second (TPS)
- **MFU**: Model FLOPS Utilization percentage
- **TFLOPS**: Achieved TFLOPS
- **Timing**: Forward/backward/optimizer step times (ms)
- **Memory**: Peak active/reserved GPU memory, allocation retries, OOMs

# Inference

The `InferencerWorker` provides distributed inference with two modes:

**`stitch_volume`**: Reconstructs full volumes from hypercube predictions via all-to-all communication.

**`save_local`**: Saves per-sample predictions locally (for detection/segmentation tasks).

Supported tasks:
- `detection` (plainDETR)
- `instance_segmentation` (MaskDINO)
- `semantic_segmentation` (mask2Former)
- `dense_prediction` (upsampling, channel split)
- `pretrain` (reconstruction)
- `feature_extractor` (feature visualization)



# Profiling

Multiple profiling tools are supported:

| Profiler | Use Case |
|----------|----------|
| **pprof** (gperftools) | CPU profiling with `@pprof_func` / `@pprof_class` decorators |
| **NVIDIA Nsys** | GPU timeline profiling via `NsysProfilerHook` |
| **PyTorch Profiler** | Operator-level profiling via `TorchProfiler` hook |
| **Memory Profiler** | PyTorch CUDA memory snapshots and `TorchMemoryStats` hook |
| **Ray Profiler** | Ray actor/task profiling via `MemoryDebugHook` |



# Configuration layout

Here's what each configuration subdirectory handles:

- **[`configs/models/`](configs/models)** - Model architectures (MAE, JEPA, plainDETR, MaskDINO, backbones, heads)
- **[`configs/datasets/`](configs/datasets)** - Dataset classes, databases, and preprocessor parameters
- **[`configs/tasks/`](configs/tasks)** - Task-specific configs (channel_split, instance_segmentation, upsample_*)
- **[`configs/optimizers/`](configs/optimizers)** - Optimizer configurations (AdamW, LAMB, Lion, Muon)
- **[`configs/schedulers/`](configs/schedulers)** - Learning rate and weight decay schedulers
- **[`configs/optimizations/`](configs/optimizations)** - Model optimizations (torch.compile, activation checkpointing)
- **[`configs/hooks/`](configs/hooks)** - Training hooks configuration
- **[`configs/checkpoint/`](configs/checkpoint)** - Checkpointing configurations
- **[`configs/deepspeed/`](configs/deepspeed)** - DeepSpeed ZeRO configurations
- **[`configs/parallelism/`](configs/parallelism)** - TorchTitan parallelism settings (TP, CP, PP, DP)
- **[`configs/clusters/`](configs/clusters)** - Cluster configurations (SLURM, LSF, local)
- **[`configs/paths/`](configs/paths)** - Path configurations for different environments (ABC, Janelia, CoreWeave)
- **[`configs/loggers/`](configs/loggers)** - Logging configurations (WandB, Local CSV)
- **[`configs/profiling/`](configs/profiling)** - Profiling configurations (pprof, nsys, torch profiler)
- **[`configs/trainer/`](configs/trainer)** - Training loop configurations
- **[`configs/evaluation/`](configs/evaluation)** - Evaluation configurations
- **[`configs/inference/`](configs/inference)** - Inference and prediction configurations
- **[`configs/tune/`](configs/tune)** - Ray Tune hyperparameter sweep configurations
- **[`configs/benchmarks/`](configs/benchmarks)** - Benchmarking configurations for throughput testing
- **[`configs/experiments/`](configs/experiments)** - Complete experiment configs and examples

# License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

   [Apache License 2.0](LICENSE)

Copyright 2025 Cell Observatory.
