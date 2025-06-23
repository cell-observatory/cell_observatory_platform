import os
import sys
import logging

from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
if (hasattr(OmegaConf, "has_resolver") and  \
    not OmegaConf.has_resolver("eval")):
    OmegaConf.register_new_resolver("eval", eval)

import ray.train.torch as raytorch

from cell_observatory_platform.utils.context import process_rank, barrier

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_dataset(cfg, transforms=None):
    rank = process_rank()
    if rank == 0:
        # initial write/setup of *.feather, *.db, etc.
        # if it does not already exist
        _ = instantiate(cfg.datasets.databases)
    barrier(device_ids=int(os.environ.get("LOCAL_RANK")))
    # all ranks read local database table and instantiate
    # database and dataset classes
    db = instantiate(cfg.datasets.databases)
    dataset = instantiate(
        cfg.datasets.dataset,
        db=db,
        transforms=transforms
    )
    return dataset


def get_dataloader(config: DictConfig):
    if config.datasets.dataloader_type == "torch":
        transforms = [instantiate(t) for t in config.datasets.transforms.transforms_list] \
                        if config.datasets.transforms.transforms_list else None

        dataset = build_dataset(config, transforms)

        if config.datasets.return_dataloader:
            collate_fn = instantiate(config.datasets.collate_fn)
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
                    num_workers=config.clusters.cpus_per_worker,
                    prefetch_factor=2,
                    persistent_workers=False,
                    sampler=DistributedSampler(train, drop_last=True)
                        if config.datasets.distributed_sampler else None,
                    # NOTE: most of worker init functionality done by Ray
                    # see https://docs.ray.io/en/latest/_modules/ray/train/torch/train_loop_utils.html
                    worker_init_fn=db_worker_init_fn,
                    drop_last=True
                )
                val = DataLoader(
                    val,
                    collate_fn=collate_fn,
                    batch_size=config.clusters.batch_size_per_gpu,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=config.clusters.cpus_per_worker,
                    prefetch_factor=2,
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
                    num_workers=config.clusters.cpus_per_worker,
                    prefetch_factor=2,
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

    else:
        # TODO: Support Ray Dataloader with heterogeneous cluster setup
        raise NotImplementedError(f"Unsupported dataloader type: {config.datasets.dataloader_type}")