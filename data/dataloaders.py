import sys
import logging
from typing import List, Optional

import ray
import torch

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

if hasattr(OmegaConf, "has_resolver") and not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)

from cell_observatory_platform.data.databases.local_database import LocalArrowDatabase
from cell_observatory_platform.data.databases.local_metadata_store import (
    MappedTable,
    TableResolver,
    fetch_object_type_names,
)
from cell_observatory_platform.data.datasets.buffers import set_buffers
from cell_observatory_platform.data.datasets.pretrain_dataset_ray import get_dataloader_ray
from cell_observatory_platform.data.datasets.schedulers import NumaNodeAffinityScheduler
from cell_observatory_platform.utils.context import (
    barrier,
    get_local_numa_nodes,
    local_rank,
    node_id,
    process_rank,
    torch_gpu_to_numa,
)

logger = logging.getLogger(__name__)


def _shape_from_stats(base_shape, layout: str, stats, selected_channel_localizations: Optional[tuple[str, ...]] = None) -> tuple[int, ...]:
    shape = list(base_shape)
    axis_to_max = {
        "T": int(stats.max_time_size),
        "Z": int(stats.max_z_size),
        "Y": int(stats.max_y_size),
        "X": int(stats.max_x_size),
        "C": int(stats.max_channel_size),
    }
    dynamic_axes = set(stats.dynamic_axes)
    for axis, max_value in axis_to_max.items():
        if axis == "C" and selected_channel_localizations is not None:
            # The EMITTED channel count, measured against the real channel arrays
            # at fetch time.
            max_value = int(stats.max_selected_channel_size)
        if axis in layout and max_value > 0:
            if dynamic_axes and axis not in dynamic_axes:
                continue
            idx = layout.index(axis)
            shape[idx] = max(int(shape[idx]), max_value)
    return tuple(shape)

def _build_dataloader_config(
    config: DictConfig,
    collate_fn,
    sample_store_desc,
    dp_degree: Optional[int],
    dp_rank: Optional[int],
    selected_channel_localizations: Optional[List[str]],
) -> dict:
    """The kwargs dict the per-epoch rebuild passes to ``get_dataloader_ray``.

    The per-epoch rebuild in ``loops.run_epoch`` consumes THIS dict, not the
    initial dataloader built in ``get_dataloader`` -- omitting a key here
    silently disables it for all of training. 
    Keep it in lockstep with the initial ``get_dataloader_ray`` call.
    """
    return {
        "cfg": config,
        "batch_size": config.clusters.batch_size_per_gpu,
        "last_batch_policy": config.datasets.last_batch_policy,
        "collate_fn": collate_fn,
        "sample_store_desc": sample_store_desc,
        "dp_degree": dp_degree,
        "dp_rank": dp_rank,
        "selected_channel_localizations": (
            list(selected_channel_localizations)
            if selected_channel_localizations is not None else None
        ),
    }


def get_dataloader(
    config: DictConfig,
    dp_degree: Optional[int] = None,
    dp_rank: Optional[int] = None,
):
    if not config.datasets.dataset._target_.endswith("PretrainDatasourceRay"):
        raise NotImplementedError(
            f"Dataset {config.datasets.dataset._target_} is not supported for dataloader building."
        )

    gpu_to_numa_map = get_local_numa_nodes(worker_numa_node=torch_gpu_to_numa(local_rank())["numa_node"])
    gpu_numa_nodes = list(gpu_to_numa_map.values()) if gpu_to_numa_map is not None else []

    db = LocalArrowDatabase(
        dbname=str(config.datasets.databases.dbname),
        dotenv_path=config.datasets.databases.dotenv_path,
        protocol=str(config.datasets.databases.protocol),
        verbose=bool(config.datasets.databases.verbose),
    )
    resolved_source = TableResolver.resolve_from_config(config, db_client=db)

    # Object-type catalog, resolved ONCE here (one row per type) and handed to
    # the collator, rather than giving every actor a DB handle.
    #
    # The annotation leaves carry object_type_id only -- deliberately, per the
    # schema ("leaves stay object_type_id + object_subtype_ids; no
    # object_type_nk"). The collator maps that id to a contiguous class index and
    # forwards the catalog into metainfo, which is how the semantic preprocessor
    # gets the class NAMES. It is batch metadata, not config: the registry splats
    # the preprocessor's config node into its constructor, so a key only one of
    # the eight preprocessors accepts would break the other seven.
    object_type_names = fetch_object_type_names(db)

    query_spec = TableResolver.build_query_spec_from_config(config, db_client=db)
    selected_channel_localizations = TableResolver.build_loader_channel_selection_from_config(
        config,
        db_client=db,
    )
    store_spec = TableResolver.build_store_spec_from_config(config)
    # Escape hatch for a cluster whose local mount differs from the catalog's
    # storage_locations.root_path.
    server_path_override = getattr(config.paths, "server_path_override", None)
    sample_store = MappedTable.create_or_attach(
        db_client=db,
        resolved=resolved_source,
        query=query_spec,
        store=store_spec,
        node_id=node_id(),
        local_rank=local_rank(),
        diagnostic_verbose=bool(getattr(config.datasets.databases, "diagnostic_verbose", False)),
        server_path_override=None if server_path_override is None else str(server_path_override),
        selected_channel_localizations=selected_channel_localizations,
    )
    sample_store_desc = sample_store.descriptor

    logger.info(
        "[DATALOADERS] source=%s rows=%s fingerprint=%s",
        sample_store_desc.sample_table.source_key,
        sample_store_desc.stats.num_rows,
        sample_store_desc.stats.ordering_fingerprint,
    )

    buffer_input_shape = _shape_from_stats(
        tuple(config.datasets.input_shape),
        config.dataset_layout_order.upper(),
        sample_store_desc.stats,
        selected_channel_localizations,
    )
    collator_input_shape = list(buffer_input_shape)

    if tuple(buffer_input_shape) != tuple(config.datasets.input_shape):
        # Deliberate two-shape scheme: the buffer/collator carry the dataset-max
        # (padded) shape; the preprocessor transforms must bring data back to
        # datasets.input_shape, enforced by _assert_input_shape_spatial.
        logger.info(
            "[DATALOADERS] buffer shape %s != datasets.input_shape %s; "
            "preprocessor transforms must resize back to input_shape.",
            tuple(buffer_input_shape),
            tuple(config.datasets.input_shape),
        )

    if local_rank() == 0:
        ray.logger.info(f"Starting NumaNodeAffinityScheduler on node {node_id()}")
        ray.logger.info(f"NUMA nodes found on this node: {gpu_numa_nodes}")
        scheduling_strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=node_id(),
            soft=False,
        )
        NumaNodeAffinityScheduler.options(
            name=f"numa_node_affinity_scheduler_node_{node_id()}",
            namespace="schedulers",
            lifetime="detached",
            max_concurrency=config.datasets.max_concurrent_calls,
            scheduling_strategy=scheduling_strategy,
        ).remote(
            policy=config.datasets.numa_node_affinity_policy,
            oversub_factor=config.datasets.numa_oversub_factor,
            node_id=node_id(),
            gpu_numa_nodes=list(set(gpu_numa_nodes)),
            gpu_to_numa_map=gpu_to_numa_map,
        )

    barrier()

    buffer_actor, _ = set_buffers(
        local_rank=local_rank(),
        global_rank=process_rank(),
        buffer_type="host_memory",
        batch_size=config.clusters.batch_size_per_gpu,
        input_shape=buffer_input_shape,
        dtype=config.storage_dtype,
        buffer_capacity=config.datasets.buffer_capacity,
        pin_numa_node=config.datasets.pin_numa_node,
        max_concurrent_calls=config.datasets.max_concurrent_calls,
        node_id=node_id(),
        pool_name="loader",
        numa_node=torch_gpu_to_numa(local_rank())["numa_node"],
    )

    # Both collators declare object_type_names, so no branching on which one the
    # config names; the pretrain collator ignores it (it builds no targets).
    collate_fn = instantiate(
        config.datasets.collate_fn,
        node_id=node_id(),
        input_shape=collator_input_shape,
        debug=config.datasets.debug,
        object_type_names=object_type_names,
    )

    dataloader_config = _build_dataloader_config(
        config=config,
        collate_fn=collate_fn,
        sample_store_desc=sample_store_desc,
        dp_degree=dp_degree,
        dp_rank=dp_rank,
        selected_channel_localizations=selected_channel_localizations,
    )
    train_dataloader, val_dataloader, _ = get_dataloader_ray(
        cfg=config,
        batch_size=config.clusters.batch_size_per_gpu,
        last_batch_policy=config.datasets.last_batch_policy,
        collate_fn=collate_fn,
        sample_store_desc=sample_store_desc,
        dp_degree=dp_degree,
        dp_rank=dp_rank,
        selected_channel_localizations=(
            list(selected_channel_localizations) if selected_channel_localizations is not None else None
        ),
    )

    return train_dataloader, val_dataloader, dataloader_config, buffer_actor, collate_fn.device_buffer, None
