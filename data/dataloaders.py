import os
import sys
import logging
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from hydra.utils import instantiate, get_method
from omegaconf import DictConfig, OmegaConf

if (hasattr(OmegaConf, "has_resolver") and not OmegaConf.has_resolver("eval")):
    OmegaConf.register_new_resolver("eval", eval)

import ray
import ray.train.torch as raytorch

from nvidia.dali.plugin.pytorch import DALIGenericIterator

from utils.context import (process_rank, 
                           barrier,
                           get_local_numa_nodes, 
                           local_rank, 
                           node_id,
                           torch_gpu_to_numa)
from data.datasets.buffers import set_buffers
from data.datasets.schedulers import NumaNodeAffinityScheduler
from data.datasets.pretrain_dataset_ray import get_dataloader_ray
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
        server_folder_path=cfg.paths.server_folder_path,
        transforms=transforms,
        max_rois=cfg.datasets.max_rois,
        max_tiles=cfg.datasets.max_tiles,
        max_hypercubes=cfg.datasets.max_hypercubes,
        hpf_list=cfg.datasets.hpf_list,
        roi_list=cfg.datasets.roi_list,
        tile_list=cfg.datasets.tile_list,
        occupancy_threshold=cfg.datasets.occupancy_threshold
    )
    return dataset


def build_dali_dataset(cfg, transforms=None):
    rank = process_rank()
    if rank == 0:
        # initialize supabase wrapper once
        db = instantiate(cfg.datasets.databases)
        dataset_len = db.hypercubes_dataframe.shape[0]
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
            server_folder_path=cfg.paths.server_folder_path,
            batch_size=cfg.clusters.batch_size_per_gpu,
            indices=train_indices,
            max_rois=cfg.datasets.max_rois,
            max_tiles=cfg.datasets.max_tiles,
            max_hypercubes=cfg.datasets.max_hypercubes,
            hpf_list=cfg.datasets.hpf_list,
            roi_list=cfg.datasets.roi_list,
            tile_list=cfg.datasets.tile_list,
            occupancy_threshold=cfg.datasets.occupancy_threshold
        )
        val_dataset = instantiate(
            cfg.datasets.dataset,
            transforms=transforms,
            hypercubes_dataframe_path=Path(cfg.datasets.databases.hypercubes_dataframe_path),
            server_folder_path=cfg.paths.server_folder_path,
            batch_size=cfg.clusters.batch_size_per_gpu,
            indices=val_indices,
            max_rois=cfg.datasets.max_rois,
            max_tiles=cfg.datasets.max_tiles,
            max_hypercubes=cfg.datasets.max_hypercubes,
            hpf_list=cfg.datasets.hpf_list,
            roi_list=cfg.datasets.roi_list,
            tile_list=cfg.datasets.tile_list,
            occupancy_threshold=cfg.datasets.occupancy_threshold
        )

        return train_dataset, val_dataset

    else:
        dataset = instantiate(
            cfg.datasets.dataset,
            transforms=transforms,
            hypercubes_dataframe_path=Path(cfg.datasets.databases.hypercubes_dataframe_path),
            server_folder_path=cfg.paths.server_folder_path,
            batch_size=cfg.clusters.batch_size_per_gpu,
            max_rois=cfg.datasets.max_rois,
            max_tiles=cfg.datasets.max_tiles,
            max_hypercubes=cfg.datasets.max_hypercubes,
            hpf_list=cfg.datasets.hpf_list,
            roi_list=cfg.datasets.roi_list,
            tile_list=cfg.datasets.tile_list,
            occupancy_threshold=cfg.datasets.occupancy_threshold
        )

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

                return train, val, None, None

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

                return dataloader, None, None, None
        else:
            return dataset

    elif config.datasets.dataset._target_.endswith("PretrainDatasetDali"):
        transforms = []
        for t in config.datasets.transforms.transforms_list:
            if isinstance(t, DictConfig):
                transforms.append(instantiate(t))
            else:
                transforms.append(get_method(t))

        train_dataset, val_dataset = build_dali_dataset(config, transforms=transforms)

        if config.datasets.split is not None:
            # DALI dataloader
            train_pipe = pretrain_dataset_pipeline(
                dataset=train_dataset,
                batch_size=config.clusters.batch_size_per_gpu,
                num_threads=config.datasets.num_workers,
                py_start_method="spawn",
                py_num_workers=config.datasets.num_workers,
                prefetch_queue_depth=config.datasets.prefetch_factor,
                exec_async=config.datasets.exec_async,
                exec_pipelined=config.datasets.exec_pipelined,
                exec_dynamic=True,
            )
            train_pipe.build()
            dali_train_loader = DALIGenericIterator(
                pipelines=train_pipe,
                output_map=["data_tensor", "get_item_time"] if train_dataset.time else ["data_tensor"],
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
                exec_async=config.datasets.exec_async,
                exec_pipelined=config.datasets.exec_pipelined,
                exec_dynamic=True,
            )
            val_pipe.build()
            dali_val_loader = DALIGenericIterator(
                pipelines=val_pipe,
                output_map=["data_tensor", "get_item_time"] if val_dataset.time else ["data_tensor"],
                size=val_dataset.full_iterations * config.clusters.batch_size_per_gpu,
                auto_reset=True,
                last_batch_policy=instantiate(config.datasets.dali_last_batch_policy)
            )

            return dali_train_loader, dali_val_loader, None, None

        else:
            # DALI dataloader
            pipe = pretrain_dataset_pipeline(
                dataset=train_dataset,
                batch_size=config.clusters.batch_size_per_gpu,
                num_threads=config.datasets.num_workers,
                py_start_method="spawn",
                py_num_workers=config.datasets.num_workers,
                prefetch_queue_depth=config.datasets.prefetch_factor,
                exec_async=config.datasets.exec_async,
                exec_pipelined=config.datasets.exec_pipelined,
                exec_dynamic=True
            )
            pipe.build()
            dali_loader = DALIGenericIterator(
                pipelines=pipe,
                output_map=["data_tensor", "get_item_time"] if train_dataset.time else ["data_tensor"],
                size=train_dataset.full_iterations * config.clusters.batch_size_per_gpu,
                auto_reset=True,
                last_batch_policy=instantiate(config.datasets.dali_last_batch_policy)
            )
            return dali_loader, None, None, None

    elif config.datasets.dataset._target_.endswith("PretrainDatasourceRay"):
        # get numa nodes for this node (gathered on local_rank 0)
        gpu_to_numa_map = get_local_numa_nodes(worker_numa_node=torch_gpu_to_numa(local_rank())["numa_node"])
        gpu_numa_nodes = list(gpu_to_numa_map.values()) if gpu_to_numa_map is not None else []

        # start numa scheduler actor on rank 0 of each node
        if local_rank() == 0:
            ray.logger.info(f"Starting NumaNodeAffinityScheduler on node {node_id()}")
            ray.logger.info(f"NUMA nodes found on this node: {gpu_numa_nodes}")
            scheduling_strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id(),
                soft=False,
            )
            actor_scheduler = NumaNodeAffinityScheduler.options(
                name=f"numa_node_affinity_scheduler_node_{node_id()}",
                namespace="schedulers", lifetime="detached",
                max_concurrency=config.datasets.max_concurrent_calls,
                scheduling_strategy=scheduling_strategy
                ).remote(
                    policy=config.datasets.numa_node_affinity_policy,
                    oversub_factor=config.datasets.numa_oversub_factor,
                    node_id=node_id(),
                    # get unique list of NUMA nodes with GPUs on this node
                    gpu_numa_nodes=list(set(gpu_numa_nodes)),
                    gpu_to_numa_map=gpu_to_numa_map
                )

        buffer_actor, buffer_cfg = set_buffers(
            local_rank=local_rank(),
            global_rank=process_rank(),
            buffer_type="host_memory",
            batch_size=config.clusters.batch_size_per_gpu,
            input_shape=config.datasets.input_shape,
            dtype=config.storage_dtype,
            buffer_capacity=config.datasets.buffer_capacity,
            pin_to_numa_node=config.datasets.pin_numa_node,
            max_concurrent_calls=config.datasets.max_concurrent_calls,
            node_id=node_id(),
            numa_node=torch_gpu_to_numa(local_rank())["numa_node"],
        )

        if isinstance(config.datasets.collate_fn, DictConfig):
            collate_fn = instantiate(config.datasets.collate_fn, node_id=node_id())
        else:
            collate_fn = get_method(config.datasets.collate_fn, node_id=node_id())

        train_dataloader, val_dataloader = get_dataloader_ray(
            cfg=config,
            batch_size=config.clusters.batch_size_per_gpu,
            drop_last=config.datasets.drop_last_policy,
            collate_fn=collate_fn
        )
        return train_dataloader, val_dataloader, buffer_actor, collate_fn.device_buffer
    
    else:
        raise NotImplementedError(
            f"Dataset {config.datasets.dataset._target_} is not supported for dataloader building."
        )