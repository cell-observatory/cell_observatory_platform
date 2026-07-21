"""
Adopted with Apache License 2.0 from
https://github.com/facebookresearch/detectron2/detectron2/utils/events.py
"""

import os
import sys
import math
import time
import logging
import warnings
import itertools
from pathlib import Path
from abc import abstractmethod
from dotenv import load_dotenv
from collections import defaultdict
from typing import Literal, Tuple, Dict, List, Any, Optional, Sequence

import wandb
import pandas as pd
from omegaconf import OmegaConf

import torch

from cell_observatory_platform.training.optimizers import OptimizersContainer
from cell_observatory_platform.training.schedulers import LRSchedulersContainer
from cell_observatory_platform.training.helpers import (
    aggregate_microbatch_losses,
    get_metric_full_name,
    METRIC_CATEGORIES,
    METRIC_CATEGORY_NAMES,
)
from cell_observatory_platform.utils.context import (
    is_torch_dist_initialized,
    process_rank,
    get_world_size,
    barrier,
    reduce_values,
)

from torchtitan.tools import utils
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.components.metrics import DeviceMemoryMonitor, build_device_memory_monitor
from cell_observatory_platform.utils.config import registers_flat_as

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EventRecorder:
    def __init__(self):
        self._iter, self._epoch, self._val_iter = 0, 0, 0
        
        self._step_scalars : dict[str, list[tuple[float, int, int]]] = defaultdict(list)
        self._epoch_scalars : dict[str, list[tuple[float, int, int]]] = defaultdict(list)

        self._tensors, self._histograms, self._traces = [], [], []
        
        self._reduce_methods: dict[str, List[str] | None] = {}


    def put_scalar(
        self,
        name,
        value,
        scope: Literal["step", "epoch"] = "step",
        reduce_method: List[str] | None = ["median"],
        category: Optional[METRIC_CATEGORIES] = None,
        prefix: Optional[str] = None,
        units: Optional[str] = None,
    ):
        full_name = get_metric_full_name(
            name=name,
            scope=scope,
            category=category,
            prefix=prefix,
            units=units,
        )
        # we need to reduce per rank and per step to get epoch averages
        # either we set this dynamically or we have a config with 
        # the reduce methods for each scalar but then we have
        # to specify each scalar we are expecting to record
        if full_name not in self._reduce_methods:
            self._reduce_methods[full_name] = reduce_method
        if scope == "step":
            self._step_scalars[full_name].append((value, self._iter, self._epoch))
        elif scope == "epoch":
            self._epoch_scalars[full_name].append((value, self._iter, self._epoch))

    def put_scalar_batch(
        self,
        name: str,
        values: Sequence[float],
        scope: Literal["step", "epoch"] = "step",
        reduce_method: List[str] | None = ["median"],
        category: Optional[METRIC_CATEGORIES] = None,
        prefix: Optional[str] = None,
        units: Optional[str] = None,
    ) -> None:
        """Append raw observations as consecutive synthetic step records."""
        if not values:
            return
        
        full_name = get_metric_full_name(
            name=name,
            scope=scope,
            category=category,
            prefix=prefix,
            units=units,
        )
        if full_name not in self._reduce_methods:
            self._reduce_methods[full_name] = reduce_method
        store = self._step_scalars if scope == "step" else self._epoch_scalars
        it, ep = self._iter, self._epoch
        next_iter = store[full_name][-1][1] + 1 if store[full_name] else 0
        if next_iter + len(values) != it:
            logger.warning("Given values do not match current iteration. Logs may be losing data and may appear inconsistent.")
        for i, v in enumerate(values):
            store[full_name].append((float(v), next_iter + i, ep))

    def put_scalars(
        self,
        scope: Literal["step", "epoch"] = "step",
        reduce_method: List[str] = ["median"],
        category: Optional[METRIC_CATEGORIES] = None,
        prefix: Optional[str] = None,
        units: Optional[str] = None,
        **kwargs
    ):
        for k, v in kwargs.items():
            assert isinstance(v, (int, float)), \
                f"Scalar value must be an int or float, got {type(v)} for key '{k}'"
            # do we really want to throw an error if the value is not finite?
            if scope == "epoch" and (math.isnan(v) or math.isinf(v)):
                warnings.warn(
                    f"Non-finite value for key '{k}': {v}. "
                )
                # raise ValueError(f"Scalar value for key '{k}' is not finite: {v}")
            self.put_scalar(
                name=k,
                value=v,
                scope=scope,
                reduce_method=reduce_method,
                category=category,
                prefix=prefix,
                units=units,
            )

    def put_tensor(self, tensor_name, tensor, tensor_metadata):
        pass

    def put_histogram(self, hist_name, hist_tensor):
        pass

    def put_trace(self, trace_name: str, trace_path: str):
        pass

    def get_step_scalars(self):
        """
        Get the dictionary of scalars recorded so far.
        Returns:
            Dict[str, List[Tuple[float, int]]]: A dictionary where keys are scalar names
            and values are lists of tuples containing scalar value and iteration number.
        """
        return self._step_scalars

    def get_epoch_scalars(self):
        """
        Get the dictionary of epoch scalars recorded so far.
        Returns:
            Dict[str, List[Tuple[float, int]]]: A dictionary where keys are scalar names
            and values are lists of tuples containing scalar value and epoch number.
        """
        return self._epoch_scalars
    
    def get_tensors(self):
        pass

    def get_histograms(self):
        pass
    
    def get_traces(self) -> Tuple[str, str]:
        pass

    def clear_scalars(self):
        for k, v in self._step_scalars.items():
            v.clear()
        
        for k, v in self._epoch_scalars.items():
            v.clear()

    def clear_tensors(self):
        pass

    def clear_histograms(self):
        pass

    def clear_traces(self):
        pass
    
    def clear(self):
        """
        Clear all recorded events.
        This method is typically called 
        after writing events to a writer.
        """
        self.clear_tensors()
        self.clear_histograms()
        self.clear_scalars()
        self.clear_traces()

    def get_reduce_op(self, name):
            return self._reduce_methods.get(name)

    def reduce_epoch_metric(self, name: str, reduce_op: Optional[str] = None) -> Optional[float]:
        """Reduce this epoch's buffered records for ``name`` to one scalar.

        Hooks (``BestMetricSaver`` / ``EarlyStopHook``) need the per-epoch value
        in ``after_validation`` — before ``PeriodicWriter`` reduces and clears.
        Mirrors the writer's reduction (pool across steps and ranks, reduce once)
        so selected == plotted. The buffer holds only the current epoch (cleared
        each ``after_epoch``). ``None`` when nothing is buffered.
        """
        records = self._epoch_scalars.get(name)
        vals = [float(v) for (v, _it, _ep) in records] if records else []

        # pool raw per-rank values then reduce once (match the writer; NOT a
        # reduce-of-per-rank-reductions)
        if is_torch_dist_initialized() and get_world_size() > 1:
            gathered: List[Optional[List[float]]] = [None] * get_world_size()
            torch.distributed.all_gather_object(gathered, vals)
            pooled = [v for part in gathered for v in (part or [])]
        else:
            pooled = vals

        if not pooled:
            return None

        ops = self.get_reduce_op(name) or ["median"]
        op = reduce_op or ops[0]
        return reduce_values(op, pooled)

    def resume(self, iter: int, epoch: int):
        """
        Resume the recorder with the given iteration and epoch.
        This is useful for resuming training from a checkpoint.
        
        Args:
            iter (int): The iteration number to resume from.
            epoch (int): The epoch number to resume from.
        """
        self._iter = iter
        self._epoch = epoch
        self._val_iter = 0


class EventWriter:
    """
    Base class for writers that obtain events from
    :class:`EventRecorder` and process them.
    """
    def __init__(self, 
                 event_recorder: EventRecorder):
        self.event_recorder = event_recorder
    
    # since each worker process has its own EventRecorder,
    # with its own sclars, we need to gather all scalars
    # from all workers
    def reduce_scalars(self):
        distributed = is_torch_dist_initialized()
        world = get_world_size()
        rank = process_rank()

        step_scalars_per_epoch, step_scalars = self._gather_scalars(
            scalars=self.event_recorder.get_step_scalars(),
            rank=rank, 
            world=world, 
            distributed=distributed,
            keep_steps_data=True
        )
        epoch_scalars, _ = self._gather_scalars(
            scalars=self.event_recorder.get_epoch_scalars(),
            rank=rank, 
            world=world, 
            distributed=distributed,
            keep_steps_data=False
        )

        if rank == 0:
            # add reduced step scalars to epoch scalars
            epoch_scalars.update(step_scalars_per_epoch)  

        return step_scalars, epoch_scalars

    def _gather_scalars(
        self,
        scalars: Dict[str, List[Tuple[float, int, int]]],
        rank: int,
        world: int,
        distributed: bool = True,
        keep_steps_data: bool = False
    ):
        if distributed and world > 1:
            gathered = [None] * world
            torch.distributed.all_gather_object(gathered, scalars)
        else:
            gathered = [scalars]

        if rank == 0:
            # {metric: {(it,ep): [vals,...]}}
            buckets = defaultdict(lambda: defaultdict(list))
            
            # {metric: {(it, ep): [val_rank0, ...]}}
            for rank, dict in enumerate(gathered):
                for name, records in dict.items():
                    for val, it, ep in records:
                        buckets[name][(it, ep)].append(val)

            # apply reductions
            merged, merged_per_step = defaultdict(list), defaultdict(list)
            for metric, rows in buckets.items():
                reduce_op_list = self.event_recorder.get_reduce_op(metric)
                # vals_per_rank = [[val_rank0_iter0, ...], [val_rank0_iter1, ...] ...]
                vals_per_rank = [v for _, v in rows.items()]
                # flatten list of lists before reduction
                vals = list(itertools.chain.from_iterable(vals_per_rank))
                for reduce_op in reduce_op_list:
                    metric_name = f"{metric}_{reduce_op}"
                    v = self._reduce(reduce_op, vals)
                    merged[metric_name].append(
                        (v, self.event_recorder._iter, self.event_recorder._epoch)
                    )
                
                if keep_steps_data:
                    for reduce_op in reduce_op_list:
                        metric_name = f"{metric}_{reduce_op}"
                        vals_per_step = [self._reduce(reduce_op, vals_rank) for vals_rank in vals_per_rank]
                        merged_per_step[metric_name] = [
                            (val, it, ep) for val, (it, ep) in zip(vals_per_step, rows.keys())
                        ]

            return merged, merged_per_step
        else:
            # other ranks return empty dict
            return {}, {}
        
    def _make_step_table(self, scalar_dict):
        rows = {}
        for metric, data in scalar_dict.items():
            for val, itr, ep in data:
                row = rows.setdefault((itr, ep), {"iter": itr, "epoch": ep})
                row[metric] = val

        if not rows:
            return 

        df = (
            pd.DataFrame.from_records(list(rows.values()))
            .sort_values(["epoch", "iter"])
            .reset_index(drop=True)
        )
        return df

    def _make_epoch_table(self, scalar_dict):
        rows = {}
        for metric, data in scalar_dict.items():
            for val, itr, ep in data:
                row = rows.setdefault((ep, itr), {"epoch": ep, "iter": itr})
                row[metric] = val

        if not rows:
            return 

        df = (
            pd.DataFrame.from_records(list(rows.values()))
            .sort_values(["epoch"])
            .reset_index(drop=True)
        )
        return df
        
    def _reduce(self, reduce_method: str, values: List[float]) -> float:
        return reduce_values(reduce_method, values)

    @abstractmethod
    def _write_scalar_impl(
        self,
        scalar_dict: Dict[str, List[Tuple[float, int, int]]],
        scope: Literal["step", "epoch"] = "step"
    ):
        pass

    @abstractmethod
    def _write_tensor_impl(self):
        pass

    @abstractmethod
    def _write_histograms_impl(self):
        pass

    @abstractmethod
    def _write_traces_impl(self):
        pass

    @abstractmethod
    def close(self):
        pass


@registers_flat_as("event_writer", "local")
class LocalEventWriter(EventWriter):
    """
    A local event writer that writes events to disk.
    """
    def __init__(
        self,
        event_recorder: EventRecorder,
        save_dir: str | Path,
        step_scalars_prefix: str,
        epoch_scalars_prefix: str,
        scalars_save_format: Literal["csv"] = "csv"
    ):
        self.event_recorder = event_recorder

        self.step_scalars_prefix = step_scalars_prefix
        self.epoch_scalars_prefix = epoch_scalars_prefix
        self.scalars_save_format = scalars_save_format

        # scalars save dir: 
        # <save_dir>/scalars/{self.scalars_prefix}.json
        self.step_scalars_savepath = Path(save_dir) / "scalars"/ \
                        f"{self.step_scalars_prefix}.{self.scalars_save_format}"
        self.epoch_scalars_savepath = Path(save_dir) / "scalars" / \
            f"{self.epoch_scalars_prefix}.{self.scalars_save_format}"
        os.makedirs(os.path.join(save_dir, "scalars"), exist_ok=True)

    def _write_scalar_impl(self, scalar_dict, scope: Literal["step", "epoch"] = "step"):
        if not scalar_dict:
            barrier()
            return

        if process_rank() == 0:
            if not scalar_dict:
                raise ValueError("No scalars to write. "
                                "Please ensure scalars are recorded before writing.")

            if self.scalars_save_format == "csv":
                if scope == "step":
                    df = self._make_step_table(scalar_dict)
                    savepath = self.step_scalars_savepath
                elif scope == "epoch":
                    logger.info(f"Epoch scalars: {scalar_dict}")
                    df = self._make_epoch_table(scalar_dict)
                    savepath = self.epoch_scalars_savepath
                df.to_csv(savepath,
                    mode="a",
                    header=not savepath.exists(),
                    index=False
                )

            else:
                raise NotImplementedError(
                    f"Unsupported scalars_save_format: {self.scalars_save_format}. Supported formats: 'csv'."
                )

        barrier()

    def _write_tensor_impl(self):
        pass

    def _write_histograms_impl(self):
        pass
    
    def _write_traces_impl(self):
        pass

    def close(self):
        pass


@registers_flat_as("event_writer", "wandb")
class WandBEventWriter(EventWriter):
    def __init__(
        self,
        event_recorder: EventRecorder,
        project: str,
        dir: str | Path,
        entity: str | None = None,
        run_name: str | None = None,
        tags: List[str] | None = None,
        resume_from: str | None = None,
        id: str | None = None,
        notes: str | None = None,
        force: bool = True,
        env_path: str | Path | None = None,
    ):
        self.event_recorder = event_recorder

        if process_rank() == 0:
            load_dotenv(env_path)
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            self.run = wandb.init(project=project,
                                    entity=entity,
                                    dir=dir,
                                    name=run_name,
                                    tags=tags,
                                    resume=resume_from,
                                    id=id,
                                    notes=notes,
                                    force=force)
            
            self.run.define_metric("iter")
            self.run.define_metric("epoch")
            catchall_step_name = get_metric_full_name(
                name="*",
                scope="step",
            )
            catchall_epoch_name = get_metric_full_name(
                name="*",
                scope="epoch",
            )
            self.run.define_metric(catchall_step_name,  step_metric="iter")
            self.run.define_metric(catchall_epoch_name, step_metric="epoch")
            for cat in METRIC_CATEGORY_NAMES:
                catchall_step_cat_name = get_metric_full_name(
                    name="*",
                    scope="step",
                    category=cat,
                )
                catchall_epoch_cat_name = get_metric_full_name(
                    name="*",
                    scope="epoch",
                    category=cat,
                )
                self.run.define_metric(catchall_step_cat_name, step_metric="iter")
                self.run.define_metric(catchall_epoch_cat_name, step_metric="epoch")
        else:
            self.run = None

    def save_config(self, cfg: Any, filename: str = "resolved_config.yaml") -> None:
        """Save the resolved run config as a W&B file."""
        if self.run is None or process_rank() != 0:
            return

        try:
            run_dir = Path(self.run.dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            config_path = run_dir / filename
            try:
                config_yaml = OmegaConf.to_yaml(cfg, resolve=True)
            except Exception:
                logger.exception("Failed to fully resolve config; saving unresolved config instead.")
                config_yaml = OmegaConf.to_yaml(cfg, resolve=False)
            config_path.write_text(config_yaml, encoding="utf-8")
            self.run.save(str(config_path), base_path=str(run_dir), policy="now")
        except Exception:
            logger.exception("Failed to save resolved config to W&B run files.")
        
    def _write_scalar_impl(
        self,
        scalar_dict,
        scope: "Literal['step','epoch']" = "step",
    ):
        # NOTE: this is an implicit rank 0 guard
        if self.run is None:
            return
        if not scalar_dict:
            raise ValueError("No scalars to write.")

        if scope == "step":
            df = self._make_step_table(scalar_dict)
            for rec in df.to_dict(orient="records"):
                it = int(rec["iter"])
                ep = int(rec.get("epoch", 0))
                payload = {
                    "iter": it,            # required so step metrics use iter
                    "epoch": ep,           # handy to see epoch with step logs
                    **{k: v for k, v in rec.items() if k not in ("iter", "epoch")},
                }
                self.run.log(payload, commit=True)

        elif scope == "epoch":
            df = self._make_epoch_table(scalar_dict)
            for rec in df.to_dict(orient="records"):
                ep = int(rec["epoch"])
                payload = {
                    "epoch": ep,           # required so epoch metrics use epoch
                    **{k: v for k, v in rec.items() if k != "epoch"},
                }
                self.run.log(payload, commit=True)             

    def _write_histograms_impl(self):
        pass

    def _write_traces_impl(self):
        pass

    def _write_tensor_impl(self):
        pass

    def close(self):
        if self.run is not None and process_rank() == 0:
            self.run.finish()


class EventWriterList(EventWriter):
    def __init__(
        self,
        writers: List[EventWriter]
    ):
        self.writers = writers
        self.event_recorder = writers[0].event_recorder
        assert all(writer.event_recorder is self.event_recorder for writer in writers), \
            "All writers must share the same EventRecorder instance."

    def write(self):
        self.write_scalars()
        self.write_tensor()
        self.write_histograms()
        self.write_traces()

    def write_scalars(self):
        step_scalars_gathered, epoch_scalars_gathered = self.reduce_scalars()
        for writer in self.writers:
            writer._write_scalar_impl(step_scalars_gathered, scope="step")
            writer._write_scalar_impl(epoch_scalars_gathered, scope="epoch")

    def write_tensor(self):
        pass

    def write_histograms(self):
        pass

    def write_traces(self):
        pass

    def close(self):
        for writer in self.writers:
            writer.close()


# adapted from: https://github.com/pytorch/torchtitan/torchtitan/components/metrics.py
class MetricsProcessor:
    """
    Metrics processor for more complicated processing of metrics.
    Args:
        cfg (dict): Job configuration.
        parallel_dims (ParallelDims): Parallel dimensions.
    """

    parallel_dims: ParallelDims
    device_memory_monitor: DeviceMemoryMonitor

    gpu_peak_flops: int
    time_last_log: float
    ntokens_since_last_log: int

    params_count: int
    num_flops_per_token: int
    optimizers: OptimizersContainer | None
    lr_schedulers: LRSchedulersContainer | None
    model_parts: list[torch.nn.Module] | None

    def __init__(
        self,
        timers: dict,
        parallel_dims: ParallelDims,
        gradient_accumulation_steps: int,
        num_flops_per_token: int = -1,
        model_param_count: int = -1,
        optimizers: OptimizersContainer | None = None,
        lr_schedulers: LRSchedulersContainer | None = None,
        model_parts: list[torch.nn.Module] | None = None,
    ):
        self.parallel_dims = parallel_dims
        self.device_memory_monitor = build_device_memory_monitor()

        self.gpu_peak_flops = utils.get_peak_flops(
            self.device_memory_monitor.device_name
        )

        self.ntokens_seen = 0
        self.data_loading_times = []
        self.ntokens_since_last_log = 0
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.optimizers = optimizers
        self.model_parts = model_parts
        self.lr_schedulers = lr_schedulers

        # These variables have to be set later as they 
        # depend on other components or model
        self.num_flops_per_token = num_flops_per_token
        self.params_count = model_param_count

        self.fwd_start, self.fwd_end = timers["fwd_start"], timers["fwd_end"]
        self.bwd_start, self.bwd_end = timers["bwd_start"], timers["bwd_end"]
        self.step_start, self.step_end = timers["step_start"], timers["step_end"]

    def process(
        self,
        data_sample: dict,
        loss_dicts: Sequence[dict],
        extra_metrics: dict[str, Any] | None = None,
    ):
        time_delta = time.perf_counter() - self.time_last_log

        tokens_batch = data_sample["metainfo"].get("tokens_per_batch", 0)
        self.ntokens_since_last_log += tokens_batch
        self.ntokens_seen += tokens_batch

        # tokens per second per device, abbreviated as tps
        tps = self.ntokens_since_last_log / (
            time_delta * self.parallel_dims.non_data_parallel_size
        )

        if self.num_flops_per_token > 0 and self.gpu_peak_flops > 0:
            # model FLOPS utilization
            # For its definition and calculation, please refer to the PaLM paper:
            # https://arxiv.org/abs/2204.02311
            mfu = 100 * self.num_flops_per_token * tps / self.gpu_peak_flops
            tflops = self.num_flops_per_token * tps / 1e12
        else:
            mfu = -1.0
            tflops = -1.0

        device_mem_stats = self.device_memory_monitor.get_peak_stats()

        torch.cuda.synchronize()
        fwd_ms = self.fwd_start.elapsed_time(self.fwd_end)
        bwd_ms = self.bwd_start.elapsed_time(self.bwd_end)
        step_ms = self.step_start.elapsed_time(self.step_end)

        metrics = {
            "data/n_tokens_seen": self.ntokens_seen,
            "data/n_tokens_since_last_log": self.ntokens_since_last_log,
            "perf/throughput(tps)": tps,
            "perf/tflops": tflops,
            "perf/mfu(%)": mfu,
            "perf/fwd_time(ms)": fwd_ms,
            "perf/bwd_time(ms)": bwd_ms,
            "perf/step_time(ms)": step_ms,
            "memory/max_active(GiB)": device_mem_stats.max_active_gib,
            "memory/max_active(%)": device_mem_stats.max_active_pct,
            "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
            "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
            "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
            "memory/num_ooms": device_mem_stats.num_ooms,
        }

        if extra_metrics is not None:
            metrics.update(extra_metrics)

        self.ntokens_since_last_log = 0
        self.data_loading_times.clear()
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

        aggregated_loss = aggregate_microbatch_losses(loss_dicts, self.gradient_accumulation_steps)
        return metrics, aggregated_loss

# --- Registry -------------------------------------------------------------- #
# LocalEventWriter / WandBEventWriter are config-selected swap points (entries in
# `config.loggers.event_writers`), so register them under the `event_writer` role.
# The class is the factory: __init__ takes flat config kwargs plus the injected
# `event_recorder=` override, exactly as the old `instantiate(cfg, event_recorder=...)`
# did — a non-mutating splat. EventWriterList and EventRecorder are single-impl
# infra (not swap points, §10.4) and stay on Hydra `instantiate`.
