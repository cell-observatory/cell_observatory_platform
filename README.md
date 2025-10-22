
# Cell Observatory Platform
**The Cell Observatory Platform** is a comprehensive framework for training and evaluating machine learning models on biological image and video datasets. Built with [PyTorch](https://pytorch.org/), accelerated and scaled with [Ray](https://www.ray.io/), model sharding using [DeepSpeed](https://www.deepspeed.ai/), and flexibly configured using [Hydra](https://hydra.cc/), it provides a modular architecture for easy customization and extension.

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
    - [1. Update experiment name](#1-update-experiment-name)
    - [2. Update your paths](#2-update-your-paths)
    - [3. Edit resource requirements](#3-edit-resource-requirements)
    - [4. Run local training job with `manager.py`](#4-run-local-training-job-with-managerpy)
    - [5. Launch multiple training jobs or Ray Tune jobs with `manager.py`](#5-launch-multiple-training-jobs-or-ray-tune-jobs-with-managerpy)
  - [Cluster setup](#cluster-setup)
    - [SLURM Setup](#slurm-setup)
    - [LSF Setup](#lsf-setup)
- [Configuration layout](#configuration-layout)
  - [Model configurations](#model-configurations)
    - [1. Select base configurations](#1-select-base-configurations)
    - [2. Override only what you need](#2-override-only-what-you-need)
  - [Adding new models](#adding-new-models)
- [Data Pipeline](#data-pipeline)
  - [Structures](#structures)
  - [Databases](#databases)
  - [Datasets](#datasets)
  - [Preprocessors](#preprocessors)
  - [MaskGenerator](#maskgenerator)
  - [Transformations](#transformations)
  - [Evaluators](#evaluators)
- [License](#license)


# Installation

## Docker [images](https://github.com/cell-observatory/cell_observatory_platform/pkgs/container/cell_observatory_platform)
Our prebuilt image with Python, Torch, and all packages installed for you.
```shell
docker pull ghcr.io/cell-observatory/cell_observatory_platform:develop_torch_25_08
```

## Clone the repository to your host system
```shell
git clone --recurse-submodules https://github.com/cell-observatory/cell_observatory_platform.git
```

To later update to the latest, greatest
```shell
git pull --recurse-submodules
```

> [!NOTE]
> If you want to run a local version of the image, see the [Dockerfile](https://github.com/cell-observatory/cell_observatory_platform/blob/main/Dockerfile)

## Setup Supabase and W&B accounts

You will need to create a Supabase and W&B account to use the platform.
Supabase can be found [Cell Observatory Database](https://supabase.com/dashboard/org/yrgvnbckfmhfyxgzzkqb), and W&B can be found [Cell Observatory Dashboard](https://wandb.ai/cell-observatory).

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
docker run --network host -u 1000 --privileged -v $(pwd):/workspace/cell_observatory_platform -w /workspace/cell_observatory_platform --env PYTHONUNBUFFERED=1 --pull missing -it --rm  --ipc host --gpus all ghcr.io/cell-observatory/cell_observatory_platform:develop_torch_25_08 bash
```

## Running docker image on a cluster via apptainer
Running an image on a cluster typically requires an apptainer version of the image, which can be generated by:

### amd64/x86_64
```shell
apptainer pull --arch amd64 --force develop_torch_25_08.sif docker://ghcr.io/cell-observatory/cell_observatory_platform:develop_torch_25_08
```

### arm64/aarch64
```shell
apptainer pull --arch arm64 --force develop_torch_25_08.sif docker://ghcr.io/cell-observatory/cell_observatory_platform:develop_torch_25_08
```

## Building a new apptainer image with a different torch version 
First, you need to build an apptainer image for torch from the containers provided by Nvidia (e.g., `25.08-py3` from this [catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags)):
``` shell
apptainer pull --arch amd64 --force pytorch_25.08-py3.sif docker://nvcr.io/nvidia/pytorch:25.08-py3
```

Then you can run the following command to build a complete image:  
``` shell
apptainer build --arch amd64 --nv --force develop_torch_25_08.sif apptainerfile.def
```

> [!IMPORTANT]
> Make sure to pass in the right argument for your system  (`amd64` or `arm64`)

# Get started

All jobs are launched through our `manager.py` script which
facilitates cluster resource allocation and Ray cluster setup. 
You may decide whether to run jobs locally or on a cluster by 
setting the `launcher_type` variable in `configs/clusters/*.yaml`. 
We show how to run jobs locally and on SLURM or LSF clusters below. 
Other cluster configurations can be added by extending the `manager.py` 
script and adding the necessary files in the `cluster` folder to support allocaing 
cluster resources (see `cluster/ray_slurm_cluster.sh` and `cluster/ray_lsf_cluster.sh`
for examples).

## Local setup

Example job configs are located in the `configs/experiments` folder. 
For local jobs, you can use our existing `configs/paths/local.yaml` 
and `configs/clusters/local.yaml` configurations. Then just edit the handful of lines below
to specify your local directory structure and resource configuration before launching the job. 
In general, if you want to extend existing functionality (to support a new 
database type, cluster type, dataloader type, etc.), you just need to implement
the necessary module (e.g. `data/databases/my_database.py`, `data/datasets/my_dataset.py`, `clusters/my_cluster.sh` to implement
a new database, dataloader, or cluster configuration respectively), and add a new Hydra configuration file
to specify under the `defaults` block in your run config.

### 1. Update experiment name
```yaml
experiment_name: test_cell_observatory_platform
wandb_project: test_cell_observatory_platform
```

### 2. Update your paths
```yaml
paths:
  # base output directory for logs, checkpoints, etc.
  outdir: ${paths.data_path}/pretrained_models/${experiment_name}
  resume_checkpointdir: null 
  pretrained_checkpointdir: null
```

### 3. Edit resource requirements
```yaml
clusters:
  batch_size: 2 # total batch size
  worker_nodes: 1 # number of worker nodes
  gpus_per_worker: 1 # number of gpus per worker node
  cpus_per_gpu: 4 # number of cpu cores per gpu
  mem_per_cpu: 16000 # ram per cpu core
```

### 4. Run local training job with `manager.py`
Run the local job using the `manager.py` script, which will pick up the Hydra config and launch the Ray job:

```bash
# Set config and then run with `manager.py`:
python cluster/manager.py --config-name=configs/test_pretrain_4d_mae_local.yaml
```

### 5. Launch multiple training jobs or Ray Tune jobs with `manager.py`
To launch multiple training jobs with `manager.py`, set the `run_type` variable to `multi_run` and define a `runs` list of 
training jobs you want to run (see `configs/benchmarks/abc/benchmark_training_4d.yaml` for an example). Note that each run config needs to specify a base configuration from which each job can override any parameters necessary. We also provide functionality to run jobs using Ray Tune's hyperparameter tuning functionality, in which case you should set `run_type` to `tune` and specify the parameters you want to sweep in the `tune` config module. For using Hydra's native sweep functionality or to run
single jobs, set `run_type` to `single_run`. 

## Cluster setup

Running a job on a cluster is very similar to the local setup. 
You need to override the `defaults` in your `my_run_config.yaml` file. 
We recommend creating `clusters/my_cluster_configuration.yaml` and
`path/my_path_configurations.yaml` to match your directory structure and
resource/cluster configurations. We have some examples below for SLURM and 
LSF in `configs/paths/*.yaml` and `configs/clusters/*.yaml`.

### SLURM Setup
```yaml
defaults:
  - clusters: abc_a100   # configs/clusters/abc_a100.yaml
  - paths: abc           # configs/paths/abc.yaml
```

### LSF Setup
```yaml
defaults:
  - clusters: janelia_h100   # configs/clusters/janelia_h100.yaml
  - paths: janelia      # configs/paths/janelia.yaml
```

# Configuration layout

Here's what each configuration subdirectory handles:

- **[`configs/models/`](configs/models)**
  - Defines complete model specifications (e.g., `JEPA` and `MAE` model architectures).
- **[`configs/hooks/`](configs/hooks)**
  - Defines training hooks for logging, checkpointing, and other custom training behaviors.
- **[`configs/datasets/`](configs/datasets)**
  - Defines dataset classes and parameters.
- **[`configs/losses/`](configs/losses)**
  - Defines loss functions used in training.
- **[`configs/transforms/`](configs/transforms)**
  - Defines data augmentation and preprocessing pipelines and parameters.
- **[`configs/optimizers/`](configs/optimizers)**
  - Defines optimizer configurations (e.g., AdamW, SGD).
- **[`configs/schedulers/`](configs/schedulers)**
  - Defines learning rate schedulers (e.g., StepLR, CosineAnnealing).
- **[`configs/checkpoint/`](configs/checkpoint)**
  - Defines checkpointing configurations for saving and loading model states.
- **[`configs/logging/`](configs/logging)**
  - Defines logging configurations for training and evaluation metrics.
- **[`configs/deepspeed/`](configs/deepspeed)**
  - Defines DeepSpeed configurations for distributed training.
- **[`configs/clusters/`](configs/clusters)**
  - Defines cluster configurations for distributed training (e.g., Ray-on-SLURM).
- **[`configs/loggers/`](configs/loggers)**
  - Defines logging configurations for experiment tracking (e.g., WandB, TensorBoard).
- **[`configs/trainer/`](configs/trainer)**
  - Defines the training loop and configurations for training models.
- **[`configs/evaluation/`](configs/evaluation)**
  - Defines evaluation configurations for assessing model performance on validation/test datasets.
- **[`configs/optimizations/`](configs/optimizations)**
  - Defines configurations for enabling model performance optimization flags/functionality
  (e.g. `torch.compile`, activation checkpointing, flags such as `PYTORCH_CUDA_ALLOC_CONF`).
- **[`configs/benchmarks/`](configs/benchmarks)**
  - Defines run configurations for benchmarking model throughput, data loading throughput, etc.
- **[`configs/experiments/`](configs/experiments)**
  - Defines run configurations for previous experiments we have run.
- **[`configs/tune/`](configs/tune)**
  - Defines Ray Tune configurations for running parameter sweeps.

## Model configurations

We use [Hydra](https://hydra.cc/) for managing experiment configurations. Hydra allows you to construct experiments by composing modular YAML files.

### 1. Select base configurations

Use the `defaults:` list to select the base YAML configurations for your experiment:

```yaml
defaults:
  - models:           jepa
  - datasets:         pretrain
  - transforms:       transforms
  - hooks:            hooks
  - _self_            # load this file’s overrides last
```

### 2. Override only what you need

Hydra handles overrides, allowing precise experiment adjustments. Scalars and lists are replaced outright, whereas dictionaries are merged recursively (only specified keys change).

Example overrides in your main config file:

```yaml
clusters:
  batch_size: 128 # total batch size

datasets:
  input_shape: [32, 128, 128, 128, 2]
  patch_shape: [4, 16, 16, 16]
  split: 0.1 # train/val split
      
models:
  _target_: models.jepa.JEPA
```


## Adding new models

Add a new YAML under the proper group, e.g. `configs/models/backbones/my_new_backbone.yaml`. To support a brand‑new model or dataset, just drop in your new small YAML and reference it in your `defaults:` block.

```yaml
defaults:
  - models/backbones: my_new_backbone
  - _self_
```

# Data Pipeline

The end-to-end flow of our data pipeline is as follows: `Database (Supabase) -> index DataFrame -> dataset -> dataloader -> preprocessor -> model`.

We first build a hypercubes table (CSV + JSON config) from Supabase, filter it (based on server path, ROI/tile/HPF, occupancy), and then choose one of three input stacks:

  -  `PyTorch` (classic Dataset/DataLoader)

  -  `DALI` (CPU external_source → GPU pipeline)

  -  `Ray Data` (custom Datasource → streaming iterator)

All three paths normalize batches to `{"data_tensor": <Tensor>, "metainfo": <dict>}` and then a `Preprocessor` enforces `dtype/device`, applies optional on-device transforms, and (optionally) generates masks for self-supervised training. 

## Structures

**`ImageList:`** A tensor container that holds images of varying sizes as a single padded tensor, storing original image sizes, layout information (`CZYX` vs `ZYXC` etc.), and providing methods for standardization and batch operations.

**`BaseDataElement:`** A base class that provides dict-like and tensor-like operations for data containers, separating metainfo (image metadata) from data fields (annotations/predictions). 

**`DataSample:`** Inherits from `BaseDataElement` that contains an `ImageList` object for data tensors and class methods `from_dict` and `to_dict` for serialization and deserialization.

## Databases

**`SupabaseDatabase:`**  Database class that constructs or queries a view of `TxZxYxXxC` hypercubes, fetches it via `ConnectorX` (Arrow path), and saves both a CSV of records and a JSON of configs. It supports:

  - Server-aware existence filters (`/clusterfs`, `/groups`, `/aws`) and optional override of `server_folder`.

  - Row limiting / ROI or tile selection / HPF filters and an occupancy filter (based on a minimum occupancy ratio) that parses per-channel occupancy data for each sample and drops low-occupancy samples.

Users that want to implement their own database class only needs to implement (with corresponding `.yaml` files) a corresponding class to generate a local CSV record with file path information for your training samples.

## Datasets

Our `get_dataloader` method in `data/dataloaders.py` is the entrypoint that instantiates the selected dataset stack, optional transforms, and builds a dataloader. We have support for the following dataloaders:

  -  `PyTorch`: `PreTrainDataset` implements a standard `DataLoader(..., num_workers, prefetch_factor, pin_memory, ...)`, and a user-defined collate function.

  -  `DALI`: `PretrainDatasetDali` builds a `DALI` pipeline of the type `CPU external source -> GPU ops`, then a `DALIGenericIterator` provides data loading functionality.

  -  `Ray`: `PretrainDatasourceRay` builds a Dataset from a custom `Datasource`, applies optional transforms, and creates an iterator with `_iter_batches`.

## Preprocessors
Our `Preprocessors` class provides a unified interface to standardize outputs from different data loader libraries and to perform any operations needed before the model step. We support the following `Preprocessors:`

**`TorchPreprocessor`**

Normalizes `PyTorch` batches to a common interface:

  -  Stacks lists, validates dtype, optionally does checks for NaN/Inf.

  -  If `with_masking`, calls `mask_generator(batch_size)` and adds
  `masks`, `context_masks`, `target_masks`, `original_patch_indices`, `channels_to_mask` to the `data_sample` metainfo.

**`DaliPreprocessor`**

Similar to `TorchPreprocessor`, but input comes as a tuple from `DALIGenericIterator`; the tensor is already on GPU.

**`RayPreprocessor`**

Similar to `TorchPreprocessor`, also allows for applying transformations.

## MaskGenerator
Generates per-batch, patch-level masks for self-supervised learning over `3D/4D` inputs with explicit time and space awareness. We currently support the following mask generation modes:

  - `BLOCKED` / `BLOCKED_TIME_ONLY` / `BLOCKED_SPACE_ONLY`: samples block sizes from specified scales, creates spatial/temporal blocks.

  - `RANDOM` / `RANDOM_SPACE_ONLY`: MAE-style `noise -> sort -> split` mask creation pipeline that splits patch-level masks into `context` and `target` sets; in `SPACE_ONLY`, the same spatial mask is repeated across time.

  - `BLOCKED_PATTERNED`: downsample time according to a provided pattern and build masks deterministically.

The `MaskGenerator` object is owned by the `Preprocessor`; when `with_masking=True`, the preprocessor attaches all mask tensors into metainfo for the model step.

## Transformations

List of methods that return a transformed tensor. **Where to apply:** If the transform is cheap, you can attach it to the dataset by attaching the necessary config to `configs/datasets/my_dataset/transforms.yaml` (will be included inside `__getitem__` in `Torch` or in `get_dataset_ray` with `Ray Data`). If it is heavy, consider applying it on device in the `Preprocessor` to avoid extra CPU cost. For `DALI` both CPU and on device operations should be specified as part of the `DALI` pipeline.

## Evaluators

**`DatasetEvaluator:`** Abstract base class defining three key methods: `reset()` for initialization, `process()` for accumulating predictions during evaluation, and `evaluate()` for computing final metrics and returning results as dictionaries.

# License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

   [Apache License 2.0](LICENSE)

Copyright 2025 Cell Observatory.
