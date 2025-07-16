import os
import sys
import logging
from pathlib import Path

from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from hydra.utils import instantiate, get_method
from omegaconf import DictConfig, OmegaConf
if (hasattr(OmegaConf, "has_resolver") and  \
    not OmegaConf.has_resolver("eval")):
    OmegaConf.register_new_resolver("eval", eval)

import ray.train.torch as raytorch

from nvidia.dali.plugin.pytorch import DALIGenericIterator

from utils.context import process_rank, barrier

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


def build_dali_dataset(cfg):
    rank = process_rank()
    if rank == 0:
        # initialize supabase wrapper once
        db = instantiate(cfg.datasets.databases)
    barrier(device_ids=int(os.environ.get("LOCAL_RANK")))

    dataset = instantiate(cfg.datasets.dataset, 
                          hypercubes_dataframe_path=Path(cfg.datasets.databases.hypercubes_dataframe_path),
                          ndim=len(list(cfg.datasets.input_shape)),
                          batch_size=cfg.clusters.batch_size_per_gpu)
    return dataset


def get_dataloader(config: DictConfig):
    if config.datasets.dataloader_type == "torch":
        transforms = [instantiate(t) for t in config.datasets.transforms.transforms_list] \
                        if config.datasets.transforms.transforms_list else None

        dataset = build_dataset(config, transforms)

        if config.datasets.return_dataloader:
            # TODO: add support for instantiate and get_method depending on if 
            #        collate_fn is a string or a 
            # callable
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
    
    elif config.datasets.dataloader_type == "dali":
        dataset = build_dali_dataset(config)

        if config.datasets.split is not None:
            # TODO: figure out how to handle splits with DALI dataloader
            raise NotImplementedError(
                f"DALI dataloader is not implemented yet with split."
            )

        else:
            # DALI dataloader
            pipe = get_method(config.datasets.dali_pipeline._target_)(
                    dataset=dataset,
                    batch_size=config.clusters.batch_size_per_gpu,
                    num_threads=config.datasets.num_workers,
                    py_start_method="spawn",
                    py_num_workers=config.datasets.num_workers,
                    prefetch_queue_depth=config.datasets.prefetch_factor,
                    exec_async=False,
                    exec_pipelined=True,
                )
            pipe.build()
            dali_loader = DALIGenericIterator(
                pipelines = pipe,
                output_map = ["data_tensor"],
                size = dataset.full_iterations * config.clusters.batch_size_per_gpu,
                auto_reset = True,
                last_batch_policy = instantiate(config.datasets.dali_last_batch_policy)
            )
            return dali_loader, None

    else:
        # TODO: Support Ray Dataloader with heterogeneous cluster setup
        raise NotImplementedError(f"Unsupported dataloader type: {config.datasets.dataloader_type}")