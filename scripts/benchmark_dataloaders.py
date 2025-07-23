import os
import gc
import sys
import time
import signal
import datetime
import logging
import cProfile
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
from itertools import product

import torch

from omegaconf import open_dict

from data.dataloaders import get_dataloader
from utils.context import is_main_process, process_rank

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def flush():
    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass


# approach for profiling workers, requires setting
# worker_init_fn in the DataLoader for Torch dataloaders
# not implemented for DALI dataloaders
# def compose_worker_init(orig_init, out_dir):
#     def _init(worker_id):
#         install_signal_profiler(out_dir)
#         if orig_init is not None: 
#             orig_init(worker_id)
#     return _init


# def install_signal_profiler(out_dir: str | os.PathLike):
#     out_dir = Path(out_dir).expanduser().resolve()
#     out_dir.mkdir(parents=True, exist_ok=True)
#     state = {"prof": None, "file": None}

#     def _start(_sig, _frm):
#         if state["prof"] is None:
#             run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             f = out_dir / f"worker_{os.getpid()}_{run_id}.prof"
#             state["file"] = f
#             state["prof"] = cProfile.Profile()
#             state["prof"].enable()

#     def _stop(_sig, _frm):
#         if state["prof"] is not None:
#             state["prof"].disable()
#             state["prof"].dump_stats(state["file"])
#             state["prof"] = None

#     signal.signal(signal.SIGUSR1, _start)
#     signal.signal(signal.SIGUSR2, _stop)


# class WorkerInit:
#     def __init__(self, out_dir, orig_init):
#         self.out_dir = out_dir
#         self.orig_init = orig_init
#     def __call__(self, worker_id):
#         install_signal_profiler(self.out_dir)
#         if self.orig_init is not None:
#             self.orig_init(worker_id)


# move data to device, similar to Ray Train's `move_to_device` 
def move_to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [move_to_device(b, device=device) for b in batch]
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device=device) for k, v in batch.items()}
    elif isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    else:
        return batch


def _measure_loader(loader, num_batches, batch_size, cfg, profile=False):

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_modules=True,
    ) if profile else nullcontext() as prof:

        time_disk2ram = 0.0
        time_ram2vram = 0.0
        total = 0.0

        it = iter(loader)

        # warmup
        for _ in range(cfg.warmup):
            next(it)

        # NOTE: only works for Torch DataLoader
        # if cfg.benchmark_worker_process:
        #     for w in loader._iterator._workers:
        #         os.kill(w.pid, signal.SIGUSR1)

        for _ in range(num_batches):
            # disk to RAM
            t0 = time.perf_counter()
            batch = next(it)
            t1 = time.perf_counter()

            # RAM to VRAM
            batch_cuda = move_to_device(batch, "cuda")
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            time_disk2ram += (t1 - t0)
            time_ram2vram += (t2 - t1)
            total += (t2 - t0)

        # if cfg.benchmark_worker_process:
        #     for w in loader._iterator._workers:
        #         os.kill(w.pid, signal.SIGUSR2)

        # average over number of batches
        # TODO: should we report average per batch
        #       or per image?
        time_disk2ram_per_batch = time_disk2ram / num_batches
        time_ram2vram_per_batch = time_ram2vram / num_batches
        total_per_batch = total / num_batches

        disk2ram_fps = num_batches / time_disk2ram
        ram2vram_fps = num_batches / time_ram2vram
        total_fps = num_batches / total

    if prof is not None:
        prof.export_chrome_trace(cfg.csv_out.replace(".csv", ".json"))

    return dict(
        time_disk2ram_per_batch = time_disk2ram_per_batch,
        time_ram2vram_per_batch = time_ram2vram_per_batch,
        total_time_per_batch = total_per_batch,
        disk2ram_fps = disk2ram_fps,
        ram2vram_fps = ram2vram_fps,
        total_fps = total_fps,
    )


def bench_dataloader(cfg):
    results = []
    benchmark_configurations = [
            (b,w,p)
            for b, w, p in product(
                list(cfg.benchmark_params.batch_size),
                list(cfg.benchmark_params.num_workers),
                list(cfg.benchmark_params.prefetch_factors))
    ]
    for (batch_size, num_workers, prefetch_factor) in benchmark_configurations:
            with open_dict(cfg):
                cfg.clusters.batch_size_per_gpu = int(batch_size)
                cfg.datasets.num_workers = int(num_workers)
                cfg.datasets.prefetch_factor = int(prefetch_factor)

            loader, _ = get_dataloader(cfg)
            num_batches = min(cfg.num_batches, len(loader) - cfg.warmup)

            logger.info("Benchmarking dataloader with config:" 
                        f"batch size {batch_size}..."
                        f"num_workers {num_workers}, "
                        f"prefetch_factor {prefetch_factor}"
            )

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            logger.info("CUDA memory before run: "
                        f"{torch.cuda.memory_allocated() / 1e9:.3f} GB")
            
            if cfg.benchmark_main_process:
                prof = cProfile.Profile()
                prof.enable()

            stats = _measure_loader(
                loader,
                num_batches=num_batches,
                batch_size=batch_size,
                cfg=cfg,
                profile=False
            )

            if cfg.benchmark_main_process:
                prof.disable()
                prof.dump_stats(cfg.csv_out.replace(".csv", ".prof"))

            if hasattr(loader, "_shutdown_workers"):
                loader._shutdown_workers()
            del loader
            torch.cuda.empty_cache()
            gc.collect()

            torch.cuda.synchronize()
            max_mem_allocated = torch.cuda.max_memory_allocated()
            logger.info("CUDA peak memory during run: "
                        f"{max_mem_allocated / 1e9:.3f} GB")

            logger.info(f"Benchmark results for config: {stats}")

            flush()

            stats.update(
                batch_size = batch_size,
                gpus_per_worker = cfg.clusters.scaling_config.resources_per_worker.GPU,
                cpus_per_worker = cfg.clusters.scaling_config.resources_per_worker.CPU,
                dataloader_num_workers = num_workers,
                dataloader_prefetch_factor = prefetch_factor,
                input_shape= cfg.datasets.input_shape,
                rank = process_rank(),
                max_memory_allocated_gb = max_mem_allocated / 1e9,  # in GB
            )

            results.append(stats)

            logger.info(f"Saving benchmark results to {cfg.csv_out}")

            # only rank-0 writes
            if is_main_process():
                csv_pth = Path(cfg.csv_out)
                csv_pth.parent.mkdir(parents=True, exist_ok=True)
                df = pd.DataFrame(results)
                df.to_csv(csv_pth, mode="a" if csv_pth.exists() else "w", header=not csv_pth.exists(), index=False)

    return stats