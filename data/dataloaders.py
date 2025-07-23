import os
import sys
import logging
from pathlib import Path

import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from hydra.utils import instantiate, get_method
from omegaconf import DictConfig, OmegaConf

if (hasattr(OmegaConf, "has_resolver") and \
        not OmegaConf.has_resolver("eval")):
    OmegaConf.register_new_resolver("eval", eval)

import ray.train.torch as raytorch

from nvidia.dali.plugin.pytorch import DALIGenericIterator

from utils.context import process_rank, barrier
from data.datasets.pretrain_dataset_dali import pretrain_dataset_pipeline

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_dataset(cfg, transforms=None):
    rank = process_rank()
    if rank == 0:
        # initialize supabase wrapper once
        db = instantiate(cfg.datasets.databases)
    barrier(device_ids=int(os.environ.get("LOCAL_RANK")))

    dataset = instantiate(
        cfg.datasets.dataset,
        hypercubes_dataframe_path=Path(cfg.datasets.databases.hypercubes_dataframe_path),
        transforms=transforms
    )
    return dataset


def build_dali_dataset(cfg, transforms=None):
    rank = process_rank()
    if rank == 0:
        # initialize supabase wrapper once
        db = instantiate(cfg.datasets.databases)
        dataset_len = len(db.hypercubes_dataframe)
    else:
        dataset_len = None

    obj = [dataset_len] if process_rank() == 0 else [None]
    # broadcast syncs
    dist.broadcast_object_list(obj, src=0)
    dataset_len = obj[0]

    if cfg.datasets.split is not None:
        val_size = round(dataset_len * cfg.datasets.split)
        train_subset, val_subset = random_split(
            range(dataset_len),
            lengths=[dataset_len - val_size, val_size]
        )
        train_indices, val_indices = train_subset.indices, val_subset.indices

        train_dataset = instantiate(
            cfg.datasets.dataset,
            transforms=transforms,
            hypercubes_dataframe_path=Path(cfg.datasets.databases.hypercubes_dataframe_path),
            batch_size=cfg.clusters.batch_size_per_gpu,
            indices=train_indices
        )
        val_dataset = instantiate(
            cfg.datasets.dataset,
            transforms=transforms,
            hypercubes_dataframe_path=Path(cfg.datasets.databases.hypercubes_dataframe_path),
            batch_size=cfg.clusters.batch_size_per_gpu,
            indices=val_indices
        )

        return train_dataset, val_dataset

    else:
        dataset = instantiate(
            cfg.datasets.dataset,
            transforms=transforms,
            hypercubes_dataframe_path=Path(cfg.datasets.databases.hypercubes_dataframe_path),
            batch_size=cfg.clusters.batch_size_per_gpu)

    return dataset, None


def get_dataloader(config: DictConfig):
    if config.datasets.split is None and config.datasets.split == 0:
        for event_writer in config.loggers.event_writers:
            if event_writer._target_.endswith("WandBEventWriter"):
                for sk, ek in zip(event_writer.step_scalar_keys, event_writer.epoch_scalar_keys):
                    assert not sk.startswith('val_'), f"WandBEventWriter can't have {sk} when {config.datasets.split=}"
                    assert not ek.startswith('val_'), f"WandBEventWriter can't have {ek} when {config.datasets.split=}"

    if config.datasets.dataset._target_.endswith("PretrainDataset"):
        transforms = [instantiate(t) for t in config.datasets.transforms.transforms_list] \
            if config.datasets.transforms.transforms_list else None

        dataset = build_dataset(config, transforms)

        if config.datasets.return_dataloader:

            if isinstance(config.datasets.collate_fn, DictConfig):
                collate_fn = instantiate(config.datasets.collate_fn)
            else:
                collate_fn = get_method(config.datasets.collate_fn)

            db_worker_init_fn = dataset.worker_init_fn

            if config.datasets.split is not None:
                val_size = round(len(dataset) * config.datasets.split)
                train, val = random_split(dataset, lengths=[len(dataset) - val_size, val_size])

                train = DataLoader(
                    train,
                    collate_fn=collate_fn,
                    batch_size=config.clusters.batch_size_per_gpu,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=config.datasets.num_workers,
                    prefetch_factor=config.datasets.prefetch_factor,
                    persistent_workers=False,
                    sampler=DistributedSampler(train, drop_last=True)
                    if config.datasets.distributed_sampler else None,
                    # NOTE: most of worker init functionality done by Ray
                    # see https://docs.ray.io/en/latest/_modules/ray/train/torch/train_loop_utils.html
                    worker_init_fn=db_worker_init_fn,
                    drop_last=True,
                )
                val = DataLoader(
                    val,
                    collate_fn=collate_fn,
                    batch_size=config.clusters.batch_size_per_gpu,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=config.datasets.num_workers,
                    prefetch_factor=config.datasets.prefetch_factor,
                    persistent_workers=False,
                    sampler=DistributedSampler(val, shuffle=False, drop_last=True)
                    if config.datasets.distributed_sampler else None,
                    worker_init_fn=db_worker_init_fn,
                    drop_last=True
                )

                if config.distributed_framework == "ray":
                    train = raytorch.prepare_data_loader(train)
                    val = raytorch.prepare_data_loader(val)

                return train, val

            else:
                dataloader = DataLoader(
                    dataset,
                    collate_fn=collate_fn,
                    batch_size=config.clusters.batch_size_per_gpu,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=config.datasets.num_workers,
                    prefetch_factor=config.datasets.prefetch_factor,
                    persistent_workers=False,
                    # handle cases where we want to run on a single GPU without distributed environment
                    sampler=DistributedSampler(dataset, drop_last=True)
                    if config.datasets.distributed_sampler else None,
                    worker_init_fn=db_worker_init_fn,
                    drop_last=True,
                )

                if config.distributed_framework == "ray":
                    dataloader = raytorch.prepare_data_loader(dataloader)

                return dataloader, None
        else:
            return dataset

    elif config.datasets.dataset._target_.endswith("PretrainDatasetDali"):
        train_dataset, val_dataset = build_dali_dataset(config, transforms=None)

        if config.datasets.split is not None:
            # DALI dataloader
            train_pipe = pretrain_dataset_pipeline(
                dataset=train_dataset,
                batch_size=config.clusters.batch_size_per_gpu,
                num_threads=config.datasets.num_workers,
                py_start_method="spawn",
                py_num_workers=config.datasets.num_workers,
                prefetch_queue_depth=config.datasets.prefetch_factor,
                exec_async=False,
                exec_pipelined=True,
                device_id=process_rank()
            )
            train_pipe.build()
            dali_train_loader = DALIGenericIterator(
                pipelines=train_pipe,
                output_map=["data_tensor", "data_time"] if train_dataset.time else ["data_tensor"],
                size=train_dataset.full_iterations * config.clusters.batch_size_per_gpu,
                auto_reset=True,
                last_batch_policy=instantiate(config.datasets.dali_last_batch_policy)
            )

            val_pipe = pretrain_dataset_pipeline(
                dataset=val_dataset,
                batch_size=config.clusters.batch_size_per_gpu,
                num_threads=config.datasets.num_workers,
                py_start_method="spawn",
                py_num_workers=config.datasets.num_workers,
                prefetch_queue_depth=config.datasets.prefetch_factor,
                exec_async=False,
                exec_pipelined=True,
                device_id=process_rank()
            )
            val_pipe.build()
            dali_val_loader = DALIGenericIterator(
                pipelines=val_pipe,
                output_map=["data_tensor", "data_time"] if val_dataset.time else ["data_tensor"],
                size=val_dataset.full_iterations * config.clusters.batch_size_per_gpu,
                auto_reset=True,
                last_batch_policy=instantiate(config.datasets.dali_last_batch_policy)
            )

            return dali_train_loader, dali_val_loader

        else:
            # DALI dataloader
            pipe = pretrain_dataset_pipeline(
                dataset=train_dataset,
                batch_size=config.clusters.batch_size_per_gpu,
                num_threads=config.datasets.num_workers,
                py_start_method="spawn",
                py_num_workers=config.datasets.num_workers,
                prefetch_queue_depth=config.datasets.prefetch_factor,
                exec_async=False,
                exec_pipelined=True,
                device_id=process_rank()
            )
            pipe.build()
            dali_loader = DALIGenericIterator(
                pipelines=pipe,
                output_map=["data_tensor", "data_time"] if train_dataset.time else ["data_tensor"],
                size=train_dataset.full_iterations * config.clusters.batch_size_per_gpu,
                auto_reset=True,
                last_batch_policy=instantiate(config.datasets.dali_last_batch_policy)
            )
            return dali_loader, None

    else:
        # TODO: Support Ray Dataloader with heterogeneous cluster setup
        raise NotImplementedError(f"Unsupported dataloader type: {config.datasets.dataset._target_}")