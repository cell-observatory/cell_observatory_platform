"""
Adopted with Apache License 2.0 from
https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/hooks.py
https://github.com/open-mmlab/mmengine/tree/main/mmengine/hooks
"""

import os
import sys
import gc
import time
import socket
import datetime
import logging
import math
import operator
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Sequence, Union

import torch
import numpy as np
import ray
from fvcore.common.timer import Timer
from ray.train import Checkpoint, report
from torch.profiler import ProfilerActivity
from torchtitan.distributed import utils as dist_utils

from cell_observatory_platform.utils.memory import (
    read_proc_meminfo,
    statvfs_usage,
    # top_dir_entries_by_size,
    torch_cuda_summary,
    process_summary,
    top_processes_rss,
    ray_resources_summary,
    now_iso,
    bytes_gb,
    ray_memory_summary,
)
from cell_observatory_platform.training.checkpoint_metadata import build_metadata
from cell_observatory_platform.training.loggers import EventWriter, WandBEventWriter
from cell_observatory_platform.training.helpers import (
    get_metric_full_name,
    log_data_sample_metrics,
    log_loss_dict,
)
from cell_observatory_platform.training.schedulers import CosineScheduler, linear_warmup_cosine_decay
from cell_observatory_platform.utils.context import is_main_process, process_rank
from cell_observatory_platform.utils.config import registers_flat_as
if TYPE_CHECKING:
    from cell_observatory_platform.training.loops import BaseTrainer, Inferencer

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HOOK_PRIORITY(Enum):
    VERY_LOW = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERY_HIGH = 4


def _wandb_run_info(trainer: Any) -> tuple[Optional[str], Optional[str]]:
    """Rank-0 W&B run id/entity if WandBEventWriter is active; else (None, None)."""
    ewl = getattr(trainer, "event_writers_list", None)
    if ewl is not None and hasattr(ewl, "writers"):
        for w in ewl.writers:
            if isinstance(w, WandBEventWriter) and getattr(w, "run", None) is not None:
                return w.run.id, getattr(w.run, "entity", None)
    return None, None


class HookBase:
    """
    Base class for hooks that can be registered with :class:`BaseTrainer`.
    """

    # a weak reference to the trainer object
    # set by the trainer when the hook is registered
    trainer: "BaseTrainer" = None

    # the priority of the hook
    # hooks with higher priority will
    # be executed earlier
    PRIORITY = HOOK_PRIORITY.MEDIUM

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def before_epoch(self):
        """
        Called before train epoch.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_backward(self, data_sample, outputs, loss_dict):
        """
        Called after the backward pass of each iteration.
        """
        pass

    def before_backward(self, data_sample, outputs, loss_dict):
        """
        Called after forward and before backward of each iteration.
        """
        pass

    def after_step(self, data_sample, outputs, loss_dict):
        """
        Called after each iteration.
        """
        pass

    def after_epoch(self):
        """
        Called after train epoch.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_validation(self):
        """
        Called before the validation loop starts.
        """
        pass

    def after_validation(self):
        """
        Called after the validation loop ends.
        """
        pass

    def before_val_step(self):
        """
        Called before each validation step.
        """
        pass

    def after_val_step(self, data_sample, outputs, loss_dict):
        """
        Called after each validation step.
        """
        pass

    def before_test(self):
        """
        Called before the test loop starts.
        """
        pass

    def after_test(self):
        """
        Called after the test loop ends.
        """
        pass

    def before_test_step(self):
        """
        Called before each test step.
        """
        pass

    def after_test_step(self, data_sample, outputs, loss_dict):
        """
        Called after each test step.
        """
        pass

    def state_dict(self):
        """
        Hooks are stateless by default, but can be made checkpointable by
        implementing `state_dict` and `load_state_dict`.
        """
        return {}


@registers_flat_as("hook", "anomaly_detector")
class AnomalyDetector(HookBase):
    """Wrap each epoch in torch.autograd.detect_anomaly."""

    def __init__(self):
        super().__init__()
        self._anom_ctx = torch.autograd.set_detect_anomaly(True, check_nan=False)
        self.loss_nans = 0

    def before_epoch(self):
        self._anom_ctx.__enter__()

    def after_epoch(self):
        self._anom_ctx.__exit__(None, None, None)

    def after_step(self, data_sample, outputs, loss_dict):
        if torch.isnan(torch.tensor(loss_dict["step_loss"])):
            self.loss_nans += 1
            logger.warning(
                f"Step loss is {loss_dict['step_loss']} \
                           for step {self.trainer._iter} in epoch {self.trainer._epoch}"
            )
            if self.loss_nans > 10:
                raise Exception(
                    f"Step loss is {loss_dict['step_loss']} \
                                for step {self.trainer._iter} in epoch {self.trainer._epoch}."
                )


@registers_flat_as("hook", "sampler_setter")
class SamplerSetter(HookBase):
    """
    A hook that sets the sampler for the trainer.
    """

    def before_epoch(self):
        if self.trainer.ray_context.get_world_size() > 1:
            self.trainer.train_dataloader.sampler.set_epoch(self.trainer._epoch)


@registers_flat_as("hook", "lr_scheduler")
class LRScheduler(HookBase):
    """
    A hook which executes a scheduler step and summarizes the LR
    for each parameter group in the optimizer.
    """

    def __init__(self, backend: str = "DEEPSPEED"):
        super().__init__()
        self.backend = backend.upper()

    def before_train(self):
        self.optimizers = self.trainer.optimizers
        self.schedulers = self.trainer.schedulers

        if self.backend == "TORCHTITAN":
            self.multi_model_opt = True
            self.update_type = "step"
            assert hasattr(self.schedulers, "schedulers"), "When using multiple optimizers, schedulers should exist."
        elif self.backend == "DEEPSPEED":
            self.multi_model_opt = False
            self.update_type = self.trainer.schedulers.update_type
        else:
            raise NotImplementedError(f"Backend {self.backend} not supported.")

        if not self.multi_model_opt:
            self._group_labels = [g.get("name", f"group{i}") for i, g in enumerate(self.optimizers.param_groups)]
            self._best_param_group_id = self.get_best_param_group_id(self.optimizers)

    @staticmethod
    def get_best_param_group_id(optimizer):
        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            # if all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    return i
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    return i

    def after_step(self, data_sample, outputs, loss_dict):
        if not self.multi_model_opt:
            lr = self.optimizers.param_groups[self._best_param_group_id]["lr"]
        else:
            lr = self.schedulers.schedulers[0].get_last_lr()[0]

        self.trainer.event_recorder.put_scalar("lr", lr)
        if self.update_type == "epoch":
            self.schedulers.step(epoch=self.trainer._epoch)
        elif self.update_type == "step":
            if self.backend == "TORCHTITAN":
                self.schedulers.step()
            elif self.backend == "DEEPSPEED":
                self.schedulers.step(self.trainer._iter)
            else:
                raise NotImplementedError(f"{self.backend=} is not supported")
        else:
            raise NotImplementedError(f"{self.update_type=} is not supported")


@registers_flat_as("hook", "iteration_timer")
class IterationTimer(HookBase):
    """
    Track the time spent for each iteration (each run_step call in the trainer).
    Print a summary at the end of training.

    Under the convention that :meth:`before_step` of all hooks should only
    take negligible amount of time, the :class:`IterationTimer` hook should be
    placed at the beginning of the list of hooks to obtain accurate timing.
    """

    # setting priority to high to ensure that
    # this hook runs early in the hook chain
    PRIORITY = HOOK_PRIORITY.HIGH

    def __init__(self, warmup_iter: int = 0):
        """
        Args:
            warmup_iter (int): the number of iterations at the beginning to exclude
                from timing.
        """
        # TODO: decide if we want to have a warmup_iter
        #       it may make more sense to remove it entirely
        #       invalidating until we decide
        self._warmup_iter = 0

        # train step timer and
        # train epoch timer
        self._step_timer = Timer()
        self._epoch_timer = Timer()

        # validation step timer and
        # validation epoch timer
        self._val_step_timer = Timer()
        self._val_timer = Timer()

        # test step timer
        # total time spent in test
        # given by the difference
        # between _start_time and current time
        self._test_timer = Timer()

        # for the total time spent not in hooks
        # different from time between
        # _start_time and current time
        # which includes time spent in hooks
        # and anywhere not in train step
        self._total_timer = Timer()

        # for the overall training time
        self._start_time = time.perf_counter()

    def before_train(self):
        self._start_time = time.perf_counter()
        self._total_timer.reset()
        self._total_timer.pause()

    def after_train(self):
        total_time = time.perf_counter() - self._start_time
        total_time_minus_hooks = self._total_timer.seconds()
        hook_time = total_time - total_time_minus_hooks

        num_iter = self.trainer._iter + 1 - self.trainer.start_iter - self._warmup_iter

        if num_iter > 0 and total_time_minus_hooks > 0:
            # speed is meaningful only after warmup
            logger.info(
                "Overall training speed: {} iterations in {} ({:.4f} s / it)".format(
                    num_iter,
                    str(datetime.timedelta(seconds=int(total_time_minus_hooks))),
                    total_time_minus_hooks / num_iter,
                )
            )

        logger.info(
            "Total training time: {} ({} on hooks/not train step)".format(
                str(datetime.timedelta(seconds=int(total_time))),
                str(datetime.timedelta(seconds=int(hook_time))),
            )
        )

    def before_step(self):
        self._step_timer.reset()
        self._total_timer.resume()

    def after_step(self, data_sample, outputs, loss_dict):
        # +1 because we're in after_step, the current step is done
        # but not yet counted
        iter_done = self.trainer._iter - self.trainer.start_iter + 1
        if iter_done > self._warmup_iter:
            sec = self._step_timer.seconds()
            self.trainer.event_recorder.put_scalars(
                step_time=sec,
                scope="step",
                category="timing",
                units="sec"
            )
            log_data_sample_metrics(self.trainer, data_sample, default_phase="training")
            log_loss_dict(self.trainer, loss_dict, phase="training")
        else:
            # reset _total_timer and _start_time
            # to avoid counting the warmup iterations
            self._start_time = time.perf_counter()
            self._total_timer.reset()

        # _total_timer only counts
        # total time in step excluding hooks
        self._total_timer.pause()

    def before_epoch(self):
        """
        Reset the timer at the beginning of each epoch.
        """
        self._epoch_timer.reset()

    def after_epoch(self):
        sec = self._epoch_timer.seconds()
        self.trainer.event_recorder.put_scalars(
            epoch_time=sec,
            scope="epoch",
            category="timing",
            units="sec"
        )

        remaining_epochs = self.trainer._max_epochs - (self.trainer._epoch + 1)
        eta = sec * remaining_epochs / 3600
        self.trainer.event_recorder.put_scalars(
            eta=eta,
            scope="epoch",
            category="timing",
            units="hrs"
        )

    def before_validation(self):
        # stop epoch timer
        # to omit counting
        # validation time
        self._val_timer.reset()
        self._epoch_timer.pause()

    def after_validation(self):
        # resume the epoch timer
        # after the validation loop
        sec = self._val_timer.seconds()
        self.trainer.event_recorder.put_scalars(
            total_time=sec,
            prefix="val",
            scope="epoch",
            category="timing",
            units="sec"
        )
        self._epoch_timer.resume()

    def before_val_step(self):
        """
        Reset the timer at the beginning of each validation step.
        """
        self._val_step_timer.reset()

    def after_val_step(self, data_sample, outputs, loss_dict):
        """Record validation-step time.

        Emitted ``epoch``-scoped, not ``step``: val steps share the frozen
        training ``_iter`` and collapse to one point/epoch, so step-scoping would
        strand sparse points on the training ``iter`` axis joined by wandb
        interpolation lines.
        """
        sec = self._val_step_timer.seconds()
        self.trainer.event_recorder.put_scalars(
            step_time=sec,
            prefix="val",
            scope="epoch",
            category="timing",
            units="sec"
        )

        # Reset the timer for the next validation step
        self._val_step_timer.reset()

        log_data_sample_metrics(self.trainer, data_sample, default_phase="validation", scope="epoch")
        log_loss_dict(self.trainer, loss_dict, phase="validation", scope="epoch")

    def before_test(self):
        """
        Reset the timer at the beginning of each test step.
        """
        self._start_time = time.perf_counter()
        self._test_timer.reset()
        self._total_timer.reset()
        self._total_timer.pause()

    def after_test(self):
        total_time = time.perf_counter() - self._start_time
        total_time_minus_hooks = self._total_timer.seconds()
        hook_time = total_time - total_time_minus_hooks

        num_iter = self.trainer._iter + 1 - self.trainer.start_iter - self._warmup_iter

        if num_iter > 0 and total_time_minus_hooks > 0:
            # speed is meaningful only after warmup
            logger.info(
                "Overall test speed: {} iterations in {} ({:.4f} s / it)".format(
                    num_iter,
                    str(datetime.timedelta(seconds=int(total_time_minus_hooks))),
                    total_time_minus_hooks / num_iter,
                )
            )

        logger.info(
            "Total test time: {} ({} on hooks/not train step)".format(
                str(datetime.timedelta(seconds=int(total_time))),
                str(datetime.timedelta(seconds=int(hook_time))),
            )
        )

    def before_test_step(self):
        self._test_timer.reset()
        self._total_timer.resume()

    def after_test_step(self, data_sample, outputs, loss_dict):
        # +1 because we're in after_step, the current step is done
        # but not yet counted
        iter_done = self.trainer._iter - self.trainer.start_iter + 1
        if iter_done > self._warmup_iter:
            sec = self._test_timer.seconds()
            self.trainer.event_recorder.put_scalars(
                step_time=sec,
                scope="step",
                prefix="test",
                category="timing",
                units="sec"
            )
            log_data_sample_metrics(self.trainer, data_sample, default_phase="testing")
            log_loss_dict(self.trainer, loss_dict, phase="testing")
        else:
            # reset _total_timer and _start_time
            # to avoid counting the warmup iterations
            self._start_time = time.perf_counter()
            self._total_timer.reset()

        # _total_timer only counts
        # total time in step excluding hooks
        self._total_timer.pause()


def _inference_batch_size_from_sample(data_sample: Optional[Dict[str, Any]]) -> int:
    """Tile/batch count for throughput; 0 if unknown."""
    if not data_sample:
        return 0
    meta = data_sample.get("metainfo")
    if meta is None:
        return 0
    if "batch_size_actual" in meta:
        return int(meta["batch_size_actual"])
    pid = meta.get("prepared_id")
    if pid is None:
        return 1
    if hasattr(pid, "__len__") and not isinstance(pid, (str, bytes)):
        return len(pid)
    return 1


@registers_flat_as("hook", "inference_metrics")
class InferenceMetricsHook(HookBase):
    """
    Pull raw metric lists from InferencerWorker, BufferManager, SaveWorker, VizWorker,
    append to EventRecorder via put_scalar_batch / put_scalar; EventWriter reduces and logs.
    Runs only when trainer has inferencer_worker (i.e. Inferencer).
    """

    PRIORITY = HOOK_PRIORITY.MEDIUM

    trainer: "Inferencer"

    _RM_TIME = ["median", "mean", "max"]
    _RM_MEAN = ["mean"]
    _RM_GAUGE = ["median"]

    def __init__(self, log_every_n_steps: int = 100):
        self._log_every_n_steps = log_every_n_steps
        self._last_logged_step = 0

    def before_test(self):
        self._inference_start_time = time.perf_counter()
        self._inference_total_samples = 0
        if hasattr(self.trainer, "inferencer_worker"):
            self.trainer.inferencer_worker.buffer_manager.enable_metrics_collection()
            ray.logger.info("[InferenceMetricsHook] Enabled metrics collection for BufferManager")
        else:
            ray.logger.info("[InferenceMetricsHook] No inferencer_worker found, skipping metrics collection")

    def before_test_step(self):
        self._step_start_time = time.perf_counter()

    def after_test_step(self, data_sample, outputs, loss_dict):
        self._step_time_ms = (time.perf_counter() - self._step_start_time) * 1000
        
        if not hasattr(self.trainer, "inferencer_worker"):
            return
        
        iw = self.trainer.inferencer_worker
        rec = self.trainer.event_recorder
        rec.put_scalar(
            "inference/step_time_ms",
            self._step_time_ms,
            scope="step",
            reduce_method=self._RM_TIME,
        )
        m = iw.get_metrics()
        for k, v in m.items():
            rec.put_scalar(
                f"inference/{k}",
                v,
                scope="step",
                reduce_method=self._RM_TIME,
            )

        self._inference_total_samples += _inference_batch_size_from_sample(data_sample)

        if self.trainer._iter - self._last_logged_step >= self._log_every_n_steps:
            self._collect_and_record()
            self._last_logged_step = self.trainer._iter

    def _collect_and_record(self) -> None:
        iw = self.trainer.inferencer_worker
        rec = self.trainer.event_recorder

        buf = iw.buffer_manager.get_metrics()
        for pool_name, pool_metrics in buf.items():
            cap = int(pool_metrics.pop("capacity", 0))
            slot_bytes = float(pool_metrics.pop("slot_bytes", 0)) / 1e9
            for metric_name, metric_value in pool_metrics.items():
                if isinstance(metric_value, list) and metric_value:
                    rec.put_scalar_batch(
                        f"buffer/{pool_name}/{metric_name}",
                        [float(x) for x in metric_value],
                        scope="step",
                        reduce_method=self._RM_TIME if "time" in metric_name else self._RM_GAUGE,
                    )
                    if metric_name == "occupied_slots" and cap > 0:
                        rec.put_scalar_batch(
                            f"buffer/{pool_name}/pct_{metric_name}",
                            [float(x)/cap*100 for x in metric_value],
                            scope="step",
                            reduce_method=self._RM_GAUGE,
                        )
                        rec.put_scalar_batch(
                            f"buffer/{pool_name}/GB_{metric_name}",
                            [float(x)*slot_bytes for x in metric_value],
                            scope="step",
                            reduce_method=self._RM_GAUGE,
                        )
        worker_metrics = {}
        if iw.save_worker is not None:
            sm = ray.get(iw.save_worker.get_metrics.remote())
            worker_metrics["save_worker"] = sm
        
        if iw.viz_worker is not None:
            vm = ray.get(iw.viz_worker.get_metrics.remote())
            worker_metrics["viz_worker"] = vm

        for worker_name, worker_metrics in worker_metrics.items():
            for metric_name, metric_value in worker_metrics.items():
                if isinstance(metric_value, list) and metric_value:
                    rec.put_scalar_batch(
                        f"{worker_name}/{metric_name}",
                        [float(x) for x in metric_value],
                        scope="step",
                        reduce_method=self._RM_TIME if "time" in metric_name else self._RM_GAUGE,
                    )
                elif np.isscalar(metric_value):
                    rec.put_scalar(
                        f"{worker_name}/{metric_name}",
                        float(metric_value),
                        scope="step",
                        reduce_method=self._RM_TIME if "time" in metric_name else self._RM_GAUGE,
                    )
                else:
                    logger.warning(f"Unknown metric value type: {type(metric_value)} for metric {metric_name} from worker {worker_name}")
        
    def after_test(self):
        if not hasattr(self.trainer, "inferencer_worker"):
            return
        self._collect_and_record()
        if getattr(self, "_inference_total_samples", 0) <= 0:
            return
        duration_s = time.perf_counter() - getattr(
            self, "_inference_start_time", time.perf_counter()
        )
        if duration_s > 0:
            samples_per_sec = self._inference_total_samples / duration_s
            self.trainer.event_recorder.put_scalars(
                scope="epoch",
                prefix="inference",
                category="timing",
                samples_per_sec=samples_per_sec,
            )


@registers_flat_as("hook", "periodic_writer")
class PeriodicWriter(HookBase):
    """
    Write events with EventWriters periodically.
    """

    def __init__(self, writers):
        """
        Args:
            writers (list[EventWriter]): a list of EventWriter instances
                to write events to.
        """
        super().__init__()
        # TODO: do we want to have an option to only write every
        #       `period` epochs?
        self._writers = writers

        # FIXME: errors related to use in cell_observatory_finetune
        # for w in self._writers.writers:
        #     assert isinstance(w, EventWriter), "All writers must be \
        #         EventWriter instances. But got: {}".format(type(w))

    def after_epoch(self):
        self._writers.write()
        self.trainer.event_recorder.clear()

    def after_train(self):
        self._writers.close()

    def after_test(self):
        """
        Write events after the test loop ends.
        """
        self._writers.write()
        self._writers.close()


@registers_flat_as("hook", "periodic_checkpointer")
class PeriodicCheckpointer(HookBase):
    """
    Checkpointing, executed every ``period`` epoch and after the last epoch.
    """

    def __init__(self, file_prefix="latest_model", backend: str = "DEEPSPEED"):
        super().__init__()
        self.backend = backend.upper()
        self.file_prefix = file_prefix

    def before_train(self):
        if self.backend == "DEEPSPEED":
            self.period = self.trainer.checkpoint_manager.save_period
        elif self.backend == "TORCHTITAN":
            self.period = self.trainer.checkpoint_save_period
        else:
            raise NotImplementedError(f"Backend {self.backend} not supported.")

    def after_epoch(self):
        """
        Checkpointing is done after each epoch.
        """
        if self.backend == "DEEPSPEED":
            if (self.trainer._epoch + 1) % self.period == 0:
                wandb_run_id, wandb_entity = _wandb_run_info(self.trainer)
                meta = build_metadata(
                    model=self.trainer.model,
                    cfg=self.trainer.cfg,
                    epoch=self.trainer._epoch + 1,
                    iter_=self.trainer._iter,
                    best_loss=self.trainer._curr_val_metric,
                    wandb_run_id=wandb_run_id,
                    wandb_entity=wandb_entity,
                )
                self.trainer.checkpoint_manager.save(
                    prefix=self.file_prefix,
                    save_epoch=self.trainer._epoch + 1,
                    save_best_loss=self.trainer._curr_val_metric,
                    save_step=self.trainer._iter,
                    metadata=meta,
                )
        elif self.backend == "TORCHTITAN":
            self.trainer.checkpoint_manager.save(curr_step=self.trainer._iter, last_step=False)
        else:
            raise NotImplementedError(f"Backend {self.backend} not supported.")

    def after_train(self):
        """
        Checkpointing is done after the last epoch.
        """
        if self.backend == "DEEPSPEED":
            if self.trainer._epoch + 1 >= self.trainer._max_epochs:
                wandb_run_id, wandb_entity = _wandb_run_info(self.trainer)
                meta = build_metadata(
                    model=self.trainer.model,
                    cfg=self.trainer.cfg,
                    epoch=self.trainer._epoch + 1,
                    iter_=self.trainer._iter,
                    best_loss=self.trainer._curr_val_metric,
                    wandb_run_id=wandb_run_id,
                    wandb_entity=wandb_entity,
                )
                self.trainer.checkpoint_manager.save(
                    prefix=self.file_prefix,
                    save_epoch=self.trainer._epoch + 1,
                    save_best_loss=self.trainer._curr_val_metric,
                    save_step=self.trainer._iter,
                    metadata=meta,
                )
        elif self.backend == "TORCHTITAN":
            self.trainer.checkpoint_manager.save(curr_step=self.trainer._iter, last_step=True)
        else:
            raise NotImplementedError(f"Backend {self.backend} not supported.")

        if self.backend == "TORCHTITAN":
            if hasattr(self.trainer, "checkpoint_manager") and self.trainer.checkpoint_manager is not None:
                self.trainer.checkpoint_manager.close()


@registers_flat_as("hook", "best_checkpointer")
class BestCheckpointer(HookBase):
    def __init__(self, checkpointdir: Union[str, Path], backend: str = "DEEPSPEED"):
        super().__init__()
        # NOTE: period is same as in PeriodicCheckpointer
        self.checkpoint_dir = Path(checkpointdir)
        self.backend = backend.upper()

    def before_train(self):
        if self.backend == "DEEPSPEED":
            self.period = self.trainer.checkpoint_manager.save_period
        elif self.backend == "TORCHTITAN":
            self.period = self.trainer.checkpoiunt_save_period
        else:
            raise NotImplementedError(f"Backend {self.backend} not supported.")

    def after_validation(self):
        if (self.trainer._epoch + 1) % self.period == 0:
            checkpoint = Checkpoint.from_directory(self.checkpoint_dir) if is_main_process() else None
        else:
            checkpoint = None

        if is_main_process():
            report(
                metrics={
                    "best_loss": self.trainer._curr_val_metric,
                    "save_step": self.trainer._iter,
                    "save_epoch": self.trainer._epoch + 1,
                },
                checkpoint=checkpoint,
            )
        else:
            report(
                metrics={
                    "best_loss": self.trainer._curr_val_metric,
                    "save_step": self.trainer._iter,
                    "save_epoch": self.trainer._epoch + 1,
                },
                checkpoint=None,
            )


@registers_flat_as("hook", "torch_memory_stats")
class TorchMemoryStats(HookBase):
    """
    Writes pytorch's cuda memory statistics periodically.
    """

    def __init__(self, step_period=50, epoch_period=1, logdir=None):
        """
        Args:
            period (int): Output stats each 'period' iterations
            max_runs (int): Stop the logging after 'max_runs'
        """
        super().__init__()
        self._step_period = step_period
        self._epoch_period = epoch_period

        self._logdir = Path(logdir) / "memory"
        self._logdir.mkdir(parents=True, exist_ok=True)

    def before_train(self):
        assert self._step_period < self.trainer.steps_per_epoch, (
            f"Step period {self._step_period} must be less than " f"steps per epoch {self.trainer.steps_per_epoch}."
        )

    def after_step(self, data_sample, outputs, loss_dict):
        if (self.trainer._iter + 1) % self._step_period == 0:
            if torch.cuda.is_available():
                max_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
                reserved_gb = torch.cuda.memory_reserved() / (1024**3)
                max_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
                allocated_gb = torch.cuda.memory_allocated() / (1024**3)

                self.trainer.event_recorder.put_scalars(
                    max_reserved_mem=max_reserved_gb,
                    reserved_mem=reserved_gb,
                    max_allocated_mem=max_allocated_gb,
                    allocated_mem=allocated_gb,
                    scope="step",
                    category="system",
                    units="GB",
                )

                torch.cuda.reset_peak_memory_stats()

    def after_epoch(self):
        if (self.trainer._epoch + 1) % self._epoch_period == 0:
            mem_log = torch.cuda.memory_summary()

            # TODO: support for saving table to
            #       wandb/tensorboard
            if is_main_process():
                with (self._logdir / f"{self.trainer._epoch}.log").open("w") as f:
                    f.write(str(mem_log))

    def after_test_step(self, data_sample, outputs, loss_dict):
        """
        Write memory stats after each test step.
        """
        if (self.trainer._iter + 1) % self._step_period == 0:
            if torch.cuda.is_available():
                max_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
                reserved_gb = torch.cuda.memory_reserved() / (1024**3)
                max_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
                allocated_gb = torch.cuda.memory_allocated() / (1024**3)

                self.trainer.event_recorder.put_scalars(
                    max_reserved_mem=max_reserved_gb,
                    reserved_mem=reserved_gb,
                    max_allocated_mem=max_allocated_gb,
                    allocated_mem=allocated_gb,
                    scope="step",
                    category="system",
                    units="GB",
                )

                torch.cuda.reset_peak_memory_stats()

    def after_test(self):
        """
        Write memory stats after the test loop ends.
        """
        mem_log = torch.cuda.memory_summary()
        if is_main_process():
            os.makedirs(self._logdir / "test", exist_ok=True)
            with (self._logdir / "test" / "memory_test.log").open("w") as f:
                f.write(str(mem_log))


@registers_flat_as("hook", "best_metric_saver")
class BestMetricSaver(HookBase):
    # Must run before BestCheckpointer (reports trainer._curr_val_metric) and
    # before PeriodicWriter (clears the epoch buffer this hook reads). Both are
    # MEDIUM, so without a higher priority this hook would run after them in
    # config order — reporting a stale/inf metric and reading a cleared buffer.
    PRIORITY = HOOK_PRIORITY.HIGH

    def __init__(
        self,
        metric_name: str,
        compare_fn: Literal["max", "min"] = "min",
        eval_after_validation: bool = True,
        period: int = 1,
    ):
        super().__init__()
        self.metric_name = metric_name
        self.compare_fn = operator.gt if compare_fn == "max" else operator.lt

        self.eval_after_validation = eval_after_validation
        self.period = period

    def _epoch_metric_key(self, prefix: str) -> str:
        return get_metric_full_name(
            name=self.metric_name,
            scope="epoch",
            category="loss",
            prefix=prefix,
        )

    def _update_best_metrics(self, val):
        if math.isnan(val) or math.isinf(val):
            return False
        self.trainer.best_metric = val
        self.trainer.best_metric_epoch = self.trainer._epoch
        self.trainer.best_metric_iter = self.trainer._iter
        return True

    def update_best_metrics(self, latest_metric_val):
        if self.compare_fn(latest_metric_val, self.trainer.best_metric):
            self._update_best_metrics(latest_metric_val)

    def after_validation(self):
        if self.eval_after_validation:
            metric_key = self._epoch_metric_key(prefix="val")
            latest_metric_val = self.trainer.event_recorder.reduce_epoch_metric(metric_key)
            if latest_metric_val is None:
                raise ValueError(
                    f"Metric {metric_key} not found in epoch logs. "
                    "Make sure to set `val_metric` in the trainer config."
                )
            self.trainer._curr_val_metric = latest_metric_val
            self.update_best_metrics(latest_metric_val)

    def after_epoch(self):
        """
        Check if the latest metric is the best so far.
        """
        # should match period of validation loop
        if (self.trainer._epoch + 1) % self.period == 0:
            if not self.eval_after_validation:
                metric_key = self._epoch_metric_key(prefix="val")
                latest_metric_val = self.trainer.event_recorder.reduce_epoch_metric(metric_key)
                if latest_metric_val is None:
                    raise ValueError(
                        f"Metric {metric_key} not found in epoch logs. "
                        "Make sure to set `val_metric` in the trainer config."
                    )
                self.trainer._curr_val_metric = latest_metric_val
                self.update_best_metrics(latest_metric_val)

    def after_test(self):
        metric_key = self._epoch_metric_key(prefix="test")
        test_metric_val = self.trainer.event_recorder.reduce_epoch_metric(metric_key)
        if test_metric_val is None:
            raise ValueError(f"Metric {metric_key} not found in test logs. ")
        self._update_best_metrics(test_metric_val)


@registers_flat_as("hook", "nsys_profiler")
class NsysProfilerHook(HookBase):
    """
    Starts Nsight Systems on step `start_iter` and stops it at `end_iter`.
    """

    def __init__(self, start_iter=50, end_iter=55, shutdown_after_profile=True):
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.shutdown_after_profile = shutdown_after_profile
        self.closed = False

    def before_step(self):
        if self.closed:
            return
        if self.trainer._iter == self.start_iter:
            # with torch.cuda.device(self.device):
            torch.cuda.cudart().cudaProfilerStart()

    def after_step(self, *args, **kwargs):
        if self.closed:
            return
        if self.trainer._iter == self.end_iter:
            # with torch.cuda.device(self.device):
            torch.cuda.cudart().cudaProfilerStop()
            self.closed = True
            if self.shutdown_after_profile:
                raise RuntimeError("Profiling complete — stopping training")


# TODO: support for saving trace to
#       wandb/tensorboard
@registers_flat_as("hook", "torch_profiler")
class TorchProfiler(HookBase):
    """
    A hook which runs `torch.profiler.profile`.

    The above example will run the profiler for iteration 10~20 and dump
    results to ``OUTPUT_DIR``. We do not profile the first few iterations
    because they are typically slower than the rest.

    The result files can be loaded in the ``chrome://tracing`` page in
    chrome browser, and the tensorboard visualizations can be visualized using
    ``tensorboard --logdir OUTPUT_DIR/log``
    """

    TorchProfilerActivities = {"CPU": torch.profiler.ProfilerActivity.CPU, "CUDA": torch.profiler.ProfilerActivity.CUDA}

    def __init__(
        self,
        output_dir,
        schedule: dict | None = None,
        activities: Sequence[ProfilerActivity | str] | None = None,
        save_tensorboard=True,
        save_memory_trace: bool = True,
        max_events_per_snapshot: int = 1000000,
        shutdown_after_profile: bool = True,
    ):
        """
        Args:
            output_dir (str): the output directory to dump tracing files.
            activities (iterable): same as in `torch.profiler.profile`.
            save_tensorboard (bool): whether to save tensorboard visualizations at output_dir/log/
        """
        super().__init__()
        self._activities = tuple(
            a if isinstance(a, ProfilerActivity) else self.TorchProfilerActivities[a.upper()]
            for a in (activities or (ProfilerActivity.CPU, ProfilerActivity.CUDA))
        )

        self._wait, self._warmup = schedule.get("wait", 0), schedule.get("warmup", 0)
        self._active, self._repeat = schedule.get("active", 1), schedule.get("repeat", 1)
        self._skip_first = schedule.get("skip_first", 0)

        self._output_dir = output_dir
        # Total steps until first trace: skip_first + (wait + warmup + active) per repeat
        self.profile_times = (
            (self._skip_first)
            + ((self._wait + self._warmup + self._active) * self._repeat)
        )  # NOTE: this is conservative: skip_first_wait could make it smaller
        self._profiler, self._closed = None, False

        os.makedirs(os.path.join(output_dir, "log"), exist_ok=True)
        self._on_trace_ready = (
            torch.profiler.tensorboard_trace_handler(
                dir_name=os.path.join(output_dir, "log"), worker_name=f"worker_{process_rank()}"
            )
            if save_tensorboard
            else None
        )

        self._save_memory_trace = save_memory_trace
        self.max_mem_events_per_snapshot = max_events_per_snapshot

        self.shutdown_after_profile = shutdown_after_profile

    def _flush_traces(self) -> None:
        """
        Called once after the profiler is stopped.  Writes extra artefacts.
        """
        # if self.trainer._profiler is None:
        #     return

        if self._save_memory_trace:
            file_prefix = f"trace_rank{process_rank()}"
            path = os.path.join(self._output_dir, f"{file_prefix}_trace")
            try:
                torch.cuda.memory._dump_snapshot(f"{path}.pickle")
            except Exception as e:
                logger.error(f"Failed to capture memory snapshot {e}")

            torch.cuda.memory._record_memory_history(enabled=None)
            
    def _flush_traces_on_oom(self, *args, **kwargs):
        self._flush_traces()

    def before_train(self):
        torch.cuda.memory._record_memory_history(max_entries=self.max_mem_events_per_snapshot)
        torch._C._cuda_attach_out_of_memory_observer(self._flush_traces_on_oom)
        self._profiler = torch.profiler.profile(
            activities=self._activities,
            schedule=torch.profiler.schedule(
                wait=self._wait, 
                warmup=self._warmup,
                active=self._active,
                repeat=self._repeat,
                skip_first=self._skip_first,
            ),
            on_trace_ready=self._on_trace_ready,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        )
        self._profiler.__enter__()

    def after_step(self, *args, **kwargs):
        if self._closed:
            return
        elif self.trainer._iter == self.profile_times - 1:
            self._profiler.stop()
            self._flush_traces()
            self._closed, self._profiler = True, None
            if self.shutdown_after_profile:
                raise RuntimeError("Profiling complete — stopping training")
        else:
            self._profiler.step()

    def after_epoch(self):
        if not self._closed:
            self._profiler.stop()
            self._flush_traces()
            self._closed, self._profiler = True, None
            if self.shutdown_after_profile:
                raise RuntimeError("Profiling complete — stopping training")


@registers_flat_as("hook", "early_stop")
class EarlyStopHook(HookBase):
    """
    A hook that stops training early if the validation metric does not improve
    for a certain number of epochs.
    """

    def __init__(
        self,
        patience,
        stopping_threshold,
        mode: Literal["min", "max"],
        metric_name: Optional[str] = None,
    ):
        super().__init__()

        self.metric_name = metric_name
        if mode == "min":
            self.compare_fn = operator.lt
        elif mode == "max":
            self.compare_fn = operator.gt
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'min' or 'max'.")

        self.wait_count = 0
        self.patience = patience
        self.stopping_threshold = stopping_threshold

        self.latest_metric_val = math.inf

    def after_validation(self):
        """
        Check if the validation metric has improved.
        If not, increment the wait count.
        If the wait count exceeds the patience, stop training.
        """
        metric_key = get_metric_full_name(
            name=self.metric_name, scope="epoch", category="loss", prefix="val",
        )
        latest_metric_val = self.trainer.event_recorder.reduce_epoch_metric(metric_key)
        if latest_metric_val is None:
            raise ValueError(
                f"Metric {metric_key} not found in epoch logs. "
                "Make sure to set `val_metric` in the trainer config."
            )

        if math.isnan(latest_metric_val) or math.isinf(latest_metric_val):
            raise ValueError(f"Validation metric {self.metric_name} is NaN or Inf. ")

        metric_diff = abs(latest_metric_val - self.latest_metric_val)

        if self.compare_fn(metric_diff, self.stopping_threshold):
            self.wait_count += 1
            logger.info(
                f"Validation metric {self.metric_name} did not improve. "
                f"Wait count: {self.wait_count}/{self.patience}."
            )
            if self.wait_count >= self.patience:
                logger.info("Early stopping triggered.")
                # setting stop_training to True
                # will stop the training loop
                # before the next epoch starts
                # see run in EpochBasedTrainer
                self.trainer.stop_training = True
        else:
            logger.info(
                f"Validation metric {self.metric_name} improved \
                            to {latest_metric_val}."
            )
            self.wait_count = 0

        self.latest_metric_val = latest_metric_val


@registers_flat_as("hook", "ema_scheduler")
class EMASchedulerHook(HookBase):
    """
    A hook that runs EMA beta update after each step.
    """

    def __init__(self, ema_start: float, ema_end: float):
        super().__init__()

        self.ema_start = ema_start
        self.ema_end = ema_end

    def before_train(self):
        self.model = self.trainer.model
        total_steps = self.trainer._max_epochs * self.trainer.steps_per_epoch
        self.ema_scheduler = (
            self.ema_start + i * (self.ema_end - self.ema_start) / total_steps for i in range(total_steps + 1)
        )

    def after_step(self, data_sample, outputs, loss_dict):
        """
        Update the EMA beta after each step.
        """
        self.model.ema_update(beta=next(self.ema_scheduler))


@registers_flat_as("hook", "teacher_temperature_scheduler")
class TeacherTemperatureSchedulerHook(HookBase):
    """
    Updates teacher temperature each step: cosine warmup from warmup_teacher_temp
    to teacher_temp over warmup_teacher_temp_epochs, then constant (DINO-style).
    """

    def __init__(
        self,
        teacher_temp_final_value: float,
        warmup_teacher_temp_value: float,
        teacher_temp_base_value: float,
        warmup_teacher_temp_epochs: int,
    ):
        super().__init__()
        
        self.warmup_teacher_temp_value = warmup_teacher_temp_value
        self.teacher_temp_base_value = teacher_temp_base_value
        self.teacher_temp_final_value = teacher_temp_final_value
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs

    def before_train(self):
        self.model = self.trainer.model
        # NOTE: in case model gets wrpped w. module, we must unwrap it
        model = getattr(self.trainer.model, "module", self.trainer.model)
        warmup_iters = int(self.warmup_teacher_temp_epochs * self.trainer.steps_per_epoch)
        self.teacher_temp_schedule = CosineScheduler(
            start_warmup_value=self.warmup_teacher_temp_value,
            base_value=self.teacher_temp_base_value,
            final_value=self.teacher_temp_final_value,
            total_iters=self.trainer._max_epochs * self.trainer.steps_per_epoch,
            warmup_iters=warmup_iters,
            freeze_iters=0,
            trunc_extra=0.0,
        )
        model.teacher_temperature = self.teacher_temp_schedule[0]

    def after_step(self, data_sample, outputs, loss_dict):
        step = self.trainer._iter + 1
        # NOTE: in case model gets wrpped w. module, we must unwrap it
        model = getattr(self.trainer.model, "module", self.trainer.model)
        model.teacher_temperature = self.teacher_temp_schedule[step]


# NOTE: see models/meta_arch/dino.py for more details
@registers_flat_as("hook", "local_loss_reweighting")
class LocalLossReweightingHook(HookBase):
    def __init__(self, start: float, peak: float, end: float, warmup_epochs: int, cosine_epochs: int):
        super().__init__()
        
        self.start = start
        self.peak = peak
        self.end = end
        self.warmup_epochs = warmup_epochs
        self.cosine_epochs = cosine_epochs
        
    def before_train(self):
        self.model = self.trainer.model
        self.local_loss_schedule = linear_warmup_cosine_decay(
            start=self.start,
            peak=self.peak,
            end=self.end,
            warmup_iterations=self.warmup_epochs * self.trainer.steps_per_epoch,
            total_iterations=self.trainer._max_epochs * self.trainer.steps_per_epoch,
        )
        self.model.local_loss_schedule = self.local_loss_schedule

@registers_flat_as("hook", "weight_decay_schedule")
class WeightDecayScheduleHook(HookBase):
    def __init__(self, backend: str = "DEEPSPEED"):
        super().__init__()
        self.backend = backend.upper()

    def before_train(self):
        self.wd_schedulers = self.trainer.wd_schedulers
        assert (
            self.wd_schedulers is not None
        ), "WeightDecayScheduleHook requires wd_schedulers to be set in the trainer."
        self.event_recorder = self.trainer.event_recorder
        if self.event_recorder is None:
            logger.warning(
                "WeightDecayScheduleHook requires event_recorder to be set in the trainer. \
                            Weight decay values will not be logged."
            )

    def after_step(self, **kwargs):
        if self.backend == "DEEPSPEED":
            # DeepSpeed performs the optimizer step at boundary
            # after that prepare WD for the next optimizer step
            # is_gradient_accumulation_boundary queries whether the current
            # micro-batch is at the boundary of gradient accumulation, and
            # thus will trigger gradient reductions and an optimizer step
            if self.trainer.model.is_gradient_accumulation_boundary():
                self.wd_schedulers.step()
                if self.event_recorder:
                    wd0 = self.trainer.optimizers.param_groups[0]["weight_decay"]
                    self.event_recorder.put_scalars(scope="step", wd=wd0)
        elif self.backend == "TORCHTITAN":
            self.wd_schedulers.step(self.trainer._iter)
            if self.event_recorder:
                wd0 = self.trainer.optimizers[0].param_groups[0]["weight_decay"]
                self.event_recorder.put_scalars(scope="step", wd=wd0)
        else:
            raise NotImplementedError(f"Backend {self.backend} not supported.")


@registers_flat_as("hook", "cuda_synchronize")
class CudaSynchronizeHook(HookBase):
    """Diagnostic hook to force CUDA stream alignment at selected train boundaries."""

    def __init__(self, sync_at: Optional[Literal["before_forward", "before_backward"]] = None):
        super().__init__()
        self.sync_at = sync_at

    def _sync(self, metric_name: str) -> None:
        if self.sync_at is None or not torch.cuda.is_available():
            return
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        self.trainer.event_recorder.put_scalars(
            scope="step",
            category="timing",
            units="sec",
            reduce_method=["median", "max", "min"],
            **{metric_name: time.perf_counter() - t0},
        )

    def before_step(self):
        if self.sync_at == "before_forward":
            self._sync("cuda_sync_before_forward_time")

    def before_backward(self, data_sample, outputs, loss_dict):
        if self.sync_at == "before_backward":
            self._sync("cuda_sync_before_backward_time")


@registers_flat_as("hook", "free_device_buffer")
class FreeDeviceBufferHook(HookBase):
    """
    A hook that frees memory buffers after each step.
    Important to use to prevent deadlocks.
    """

    def before_train(self):
        self.device_buffer = self.trainer.device_buffer
        self.with_grad_accumulation = self.trainer.with_grad_accumulation

    def before_test(self):
        self.device_buffer = self.trainer.device_buffer
        self.with_grad_accumulation = self.trainer.with_grad_accumulation

    def after_step(self, **kwargs):
        if not self.with_grad_accumulation:
            device_buffer_idx = kwargs["data_sample"]["metainfo"]["device_buffer_idx"]
            self.device_buffer.put_free(device_buffer_idx)

    def after_test_step(self, data_sample, outputs, loss_dict):
        if not self.with_grad_accumulation:
            device_buffer_idx = data_sample["metainfo"]["device_buffer_idx"]
            self.device_buffer.put_free(device_buffer_idx)

    def after_val_step(self, data_sample, outputs, loss_dict):
        if not self.with_grad_accumulation:
            device_buffer_idx = data_sample["metainfo"]["device_buffer_idx"]
            self.device_buffer.put_free(device_buffer_idx)

    def after_backward(self, **kwargs):
        if self.with_grad_accumulation:
            device_buffer_idx = kwargs["data_sample"]["metainfo"]["device_buffer_idx"]
            self.device_buffer.put_free(device_buffer_idx)


@registers_flat_as("hook", "adjust_timeout")
class AdjustTimeoutHook(HookBase):
    """
    A hook that adjusts the training timeout for distributed processes.
    """

    def __init__(self, world_mesh, timeout: int):
        super().__init__()
        self.world_mesh = world_mesh
        self.train_timeout_seconds = timeout

    def before_step(self):
        if self.trainer._iter == 1:
            dist_utils.set_pg_timeouts(
                timeout=datetime.timedelta(seconds=self.train_timeout_seconds),
                world_mesh=self.world_mesh,
            )


@registers_flat_as("hook", "memory_debug")
class MemoryDebugHook(HookBase):
    PRIORITY = HOOK_PRIORITY.VERY_LOW

    def __init__(
        self,
        log_path: str,
        shm_top_n: int = 30,
        include_top_processes: bool = True,
        include_ray: bool = True,
        include_cuda: bool = True,
        dump_before_train: bool = True,
        dump_after_train: bool = True,
        dump_after_validation: bool = False,
    ):
        super().__init__()
        self.log_path = log_path

        self.shm_top_n = shm_top_n

        self.include_top_processes = include_top_processes
        self.include_ray = include_ray
        self.include_cuda = include_cuda

        self.dump_before_train = dump_before_train
        self.dump_after_train = dump_after_train
        self.dump_after_validation = dump_after_validation

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def before_train(self):
        if not self.dump_before_train:
            return
        if not is_main_process():
            return
        self.dump(step=getattr(self.trainer, "_iter", None), tag=f"before_train epoch={self.trainer._epoch}")

    def after_epoch(self):
        if not is_main_process():
            return
        self.dump(step=self.trainer._iter, tag=f"after_epoch epoch={self.trainer._epoch}")

    def after_validation(self):
        if not self.dump_after_validation:
            return
        if not is_main_process():
            return
        self.dump(step=self.trainer._iter, tag=f"after_validation epoch={self.trainer._epoch}")

    def after_train(self):
        if not self.dump_after_train:
            return
        if not is_main_process():
            return
        self.dump(step=getattr(self.trainer, "_iter", None), tag=f"after_train epoch={self.trainer._epoch}")

    def dump(self, step: Optional[int] = None, tag: str = "") -> None:
        host = socket.gethostname()

        keys = [
            "MemTotal", "MemFree", "MemAvailable",
            "Buffers", "Cached", "Shmem",
            "Slab", "SReclaimable",
            "Unevictable", "Mlocked",
            "SwapTotal", "SwapFree",
        ]
        mi = read_proc_meminfo(keys)

        shm_total, shm_used, shm_free = statvfs_usage("/dev/shm")
        # shm_top = top_dir_entries_by_size("/dev/shm", n=self.shm_top_n)

        parts = []
        parts.append(
            f"===== MEM SNAPSHOT { now_iso() } host={host} pid={os.getpid()} rank={0} "
            f"step={step} tag={tag} ====="
        )

        parts.append("[process]")
        parts.append(process_summary())

        parts.append("\n[/proc/meminfo]")
        for k in keys:
            if k in mi:
                parts.append(f"{k:>12}: {bytes_gb(mi[k])}")
            else:
                parts.append(f"{k:>12}: NA")

        parts.append("\n[/dev/shm]")
        parts.append(f"total={bytes_gb(shm_total)} used={bytes_gb(shm_used)} free={bytes_gb(shm_free)}")
        # parts.append("\n[/dev/shm top entries by size]")
        # parts.append(shm_top)

        if self.include_ray:
            parts.append("\n[ray resources]")
            parts.append(ray_resources_summary())

            parts.append("\n[ray object store / memory summary]")
            if self.trainer._epoch > 40 and self.trainer._epoch % 50 == 0:
                parts.append(ray_memory_summary(stats_only=False))
            else:
                parts.append(ray_memory_summary(stats_only=True))

        if self.include_top_processes:
            parts.append("\n[top processes by RSS]")
            parts.append(top_processes_rss(n=15))

        if self.include_cuda:
            parts.append("\n[torch cuda]")
            parts.append(torch_cuda_summary())

        parts.append("\n")

        blob = "\n".join(parts)

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(blob)
            if not blob.endswith("\n"):
                f.write("\n")
            f.flush()
            os.fsync(f.fileno())


@registers_flat_as("hook", "garbage_collection")
class GarbageCollectionHook(HookBase):
    """
    GC control Hook.
    - Disables Python's automatic GC (optional)
    - Forces periodic gc.collect() in sync across ranks
    - Optional debug: runs extra GC and warns about tensor cycles (rank0 only)
    """

    PRIORITY = HOOK_PRIORITY.VERY_LOW

    def __init__(
        self,
        gc_freq: int = 1000,
        debug: bool = False,
        disable_auto_gc: bool = True,
        generation: int = 1,
        run_on: Literal["step", "epoch"] = "step",
        only_rank0: bool = False,
        initial_collect: bool = True,
        reenable_on_end: bool = False,
    ):
        super().__init__()
        assert gc_freq > 0, "gc_freq must be a positive integer"
        assert generation in (0, 1, 2), "generation must be 0, 1, or 2"
        assert run_on in ("step", "epoch"), "run_on must be 'step' or 'epoch'"

        self.gc_freq = int(gc_freq)
        self.debug = bool(debug)
        self.disable_auto_gc = bool(disable_auto_gc)
        self.generation = int(generation)
        self.run_on = run_on
        self.only_rank0 = bool(only_rank0)
        self.initial_collect = bool(initial_collect)
        self.reenable_on_end = bool(reenable_on_end)

        self._auto_gc_was_enabled: Optional[bool] = None

    def _should_run_this_rank(self) -> bool:
        if not self.only_rank0:
            return True
        return is_main_process()

    @staticmethod
    def collect(reason: str, generation: int = 1):
        begin = time.monotonic()
        gc.collect(generation)
        logger.info("[GC] %s took %.2f seconds", reason, time.monotonic() - begin)

    def before_train(self):
        if not self._should_run_this_rank():
            return

        self._auto_gc_was_enabled = gc.isenabled()
        if self.disable_auto_gc:
            gc.disable()

        if self.initial_collect:
            self.collect("Initial GC collection", generation=self.generation)

        if self.debug:
            try:
                from torch.utils.viz._cycles import warn_tensor_cycles

                if is_main_process():
                    warn_tensor_cycles()
            except Exception as e:
                logger.info("[GC] warn_tensor_cycles unavailable (%s)", e)

    def after_step(self, *args, **kwargs):
        if self.run_on != "step":
            return
        if not self._should_run_this_rank():
            return

        step_count = int(getattr(self.trainer, "_iter", 0)) + 1

        if self.debug:
            self.collect("Force GC to perform collection to obtain debug information", generation=2)
            gc.collect()
            return

        if step_count > 1 and (step_count % self.gc_freq == 0):
            self.collect("Performing periodic GC collection", generation=self.generation)

    def after_epoch(self, *args, **kwargs):
        if self.run_on != "epoch":
            return
        if not self._should_run_this_rank():
            return

        epoch_count = int(getattr(self.trainer, "_epoch", 0)) + 1

        if self.debug:
            self.collect("Force GC (epoch) to perform collection to obtain debug information", generation=2)
            gc.collect()
            return

        if epoch_count > 0 and (epoch_count % self.gc_freq == 0):
            self.collect("Performing periodic GC (epoch) collection", generation=self.generation)

    def after_train(self):
        if not self._should_run_this_rank():
            return
        if self.reenable_on_end and self._auto_gc_was_enabled:
            gc.enable()
