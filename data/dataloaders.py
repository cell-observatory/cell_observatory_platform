import os
import sys
import logging
from pathlib import Path

import torch
from hydra.utils import get_method, instantiate
from omegaconf import DictConfig, OmegaConf

if hasattr(OmegaConf, "has_resolver") and not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)

import ray

from cell_observatory_platform.data.datasets.buffers import set_buffers
from cell_observatory_platform.data.datasets.schedulers import NumaNodeAffinityScheduler
from cell_observatory_platform.data.datasets.pretrain_dataset_ray import get_dataloader_ray
from cell_observatory_platform.utils.context import (
    barrier,
    get_local_numa_nodes,
    local_rank,
    node_id,
    process_rank,
    torch_gpu_to_numa,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_dataloader(config: DictConfig):
    if config.datasets.split is None and config.datasets.split == 0:
        for event_writer in config.loggers.event_writers:
            if event_writer._target_.endswith("WandBEventWriter"):
                for sk, ek in zip(event_writer.step_scalar_keys, event_writer.epoch_scalar_keys):
                    assert not sk.startswith("val_"), f"WandBEventWriter can't have {sk} when {config.datasets.split=}"
                    assert not ek.startswith("val_"), f"WandBEventWriter can't have {ek} when {config.datasets.split=}"

    if config.datasets.dataset._target_.endswith("PretrainDatasourceRay"):
        # get numa nodes for this node (gathered on local_rank 0)
        gpu_to_numa_map = get_local_numa_nodes(worker_numa_node=torch_gpu_to_numa(local_rank())["numa_node"])
        gpu_numa_nodes = list(gpu_to_numa_map.values()) if gpu_to_numa_map is not None else []

        # TODO: we should consider instantiating the database in a
        #       partitioned way so each ray actor only loads a subset of the data
        #       to allow for better scaling when we move towards billion sample datasets
        #       this is probably the best entrypoint for that change
        db = None
        buffer_input_shape = tuple(config.datasets.input_shape)

        try:
            tile_mode = (
                hasattr(config.datasets.databases, "with_hypercubes_dataframe")
                and not config.datasets.databases.with_hypercubes_dataframe
            )
        except Exception:
            tile_mode = False

        if tile_mode:
            db = instantiate(config.datasets.databases)
            df = db.hypercubes_dataframe

            if not {"z_size", "y_size", "x_size"}.issubset(df.columns):
                raise ValueError("Tile-level training requires z_size, y_size, x_size columns in tiles dataframe.")

            max_z = int(df["z_size"].max())
            max_y = int(df["y_size"].max())
            max_x = int(df["x_size"].max())

            layout = config.dataset_layout_order.upper()
            ax2idx = {ax: i for i, ax in enumerate(layout)}

            buffer_input_shape_list = list(buffer_input_shape)

            if "Z" in ax2idx:
                buffer_input_shape_list[ax2idx["Z"]] = max_z
            if "Y" in ax2idx:
                buffer_input_shape_list[ax2idx["Y"]] = max_y
            if "X" in ax2idx:
                buffer_input_shape_list[ax2idx["X"]] = max_x

            buffer_input_shape = tuple(buffer_input_shape_list)

            logger.info(
                f"[DATALOADERS] Using buffer_input_shape={buffer_input_shape} "
                f"[DATALOADERS] (max tile sizes: z={max_z}, y={max_y}, x={max_x})"
            )

            if config.datasets.collate_fn.with_resize:
                collator_input_shape = list(config.datasets.input_shape)
            else:
                collator_input_shape = list(buffer_input_shape)
        else:
            collator_input_shape = list(config.datasets.input_shape)

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
                namespace="schedulers",
                lifetime="detached",
                max_concurrency=config.datasets.max_concurrent_calls,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                policy=config.datasets.numa_node_affinity_policy,
                oversub_factor=config.datasets.numa_oversub_factor,
                node_id=node_id(),
                # get unique list of NUMA nodes with GPUs on this node
                gpu_numa_nodes=list(set(gpu_numa_nodes)),
                gpu_to_numa_map=gpu_to_numa_map,
            )

        barrier()

        buffer_actor, buffer_cfg = set_buffers(
            local_rank=local_rank(),
            global_rank=process_rank(),
            buffer_type="host_memory",
            batch_size=config.clusters.batch_size_per_gpu,
            input_shape=buffer_input_shape,
            dtype=config.storage_dtype,
            buffer_capacity=config.datasets.buffer_capacity,
            pin_to_numa_node=config.datasets.pin_numa_node,
            max_concurrent_calls=config.datasets.max_concurrent_calls,
            node_id=node_id(),
            numa_node=torch_gpu_to_numa(local_rank())["numa_node"],
        )

        collate_fn = instantiate(
            config.datasets.collate_fn, node_id=node_id(), input_shape=collator_input_shape, debug=config.datasets.debug
        )

        train_dataloader, val_dataloader, database_df = get_dataloader_ray(
            cfg=config,
            batch_size=config.clusters.batch_size_per_gpu,
            drop_last=config.datasets.drop_last_policy,
            collate_fn=collate_fn,
            database=db,
        )
        return train_dataloader, val_dataloader, buffer_actor, collate_fn.device_buffer, database_df

    else:
        raise NotImplementedError(
            f"Dataset {config.datasets.dataset._target_} is not supported for dataloader building."
        )
