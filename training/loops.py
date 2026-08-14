"""
Adopted with Apache License 2.0 from
https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/train_loop.py
https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
"""

import copy
import math
import os
import time
import logging
import weakref
from datetime import timedelta
from pathlib import Path
from typing import Any, List, Optional, Sequence, Dict

from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import get_class, instantiate

if not OmegaConf.has_resolver("now"):
    OmegaConf.register_new_resolver("now", lambda fmt: time.strftime(fmt))

import torch
# NOTE: `deepspeed` is imported lazily inside EpochBasedTrainer.__init__.
# Save/viz/buffer Ray actors import modules that (transitively) import this
# module's package; a module-scope deepspeed import cost every actor ~9s.
from ray.train import get_context
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import ray

from cell_observatory_platform.training.helpers import (
    enable_optimizations,
    get_masked_input_data,
    get_steps_per_epoch,
    initial_best_metric,
    kill_stale_actor,
    resume_run,
    set_global_seed,
    summarize_model,
    apply_compile,
    apply_activation_checkpointing,
    configure_torch_comm_env,
    get_model_optimizations_node,
    get_metric_full_name,
)
from cell_observatory_platform.training.hooks import HookBase
from cell_observatory_platform.data.dataloaders import get_dataloader
from cell_observatory_platform.training.loggers import EventRecorder, MetricsProcessor
from cell_observatory_platform.training.optimizers import build_optimizers, get_optimizer
# quantize -> AC -> compile -> fully_shard for the torch-native trainer;
# imports torch only at module scope (torchao/torchtitan stay lazy)
from cell_observatory_platform.parallelism.parallelize import parallelize
from cell_observatory_platform.data.datasets.pretrain_dataset_ray import get_dataloader_ray
from cell_observatory_platform.utils.context import (
    inference_context,
    is_main_process,
    process_rank,
    get_world_size,
    local_rank,
    node_id,
    torch_gpu_to_numa,
)
from cell_observatory_platform.inference.inferencer import InferencerWorker
from cell_observatory_platform.training.schedulers import (
    build_lr_schedulers,
    build_wd_schedulers,
    get_param_groups,
    get_schedulers,
)
from cell_observatory_platform.data.datasets.buffers import BufferManager, init_output_memory_pools
from cell_observatory_platform.inference.saver import SaveWorker
from cell_observatory_platform.inference.visualizer import VizWorker

from cell_observatory_platform.utils.registry import REGISTRY
import cell_observatory_platform.utils._register  # noqa: F401  walk-imports + populates REGISTRY

logger = logging.getLogger("ray")
logger.setLevel(logging.INFO)

# silence broken logging call in Ray internals to prevent
# checkpoint saving from failing
logging.getLogger("ray.train._internal.checkpoint_manager").setLevel(logging.INFO)


def _ensure_full_path(config: DictConfig) -> DictConfig:
    """Fix any relative imports in _target_ fields by prefixing with `cell_observatory_platform.`"""
    if "_target_" in config and config._target_ and not config._target_.startswith("cell_observatory_platform."):
        config._target_ = f"cell_observatory_platform.{config._target_}"
    return config


# Ray train wrapper entry point
def train_loop_per_worker(config):
    if not isinstance(config, DictConfig):
        # Ray Tune ships train_loop_config as a plain dict (samplers resolved
        # to concrete values); the non-tune path ships the DictConfig itself.
        config = OmegaConf.create(config)
    trainer_cls = get_class(config.trainer)
    trainer_per_worker = trainer_cls(config)

    if config.job_type == "train":
        trainer_per_worker.run()
    elif config.job_type == "test":
        trainer_per_worker.test()
    elif config.job_type == "predict":
        trainer_per_worker.predict()
    else:
        raise ValueError(f"Unknown job type: {config.job_type}. "
                         f"Expected 'train', 'test' or 'predict', got '{config.job_type}'.")
    
    return {"best_metric": trainer_per_worker.best_metric 
            if hasattr(trainer_per_worker, 'best_metric') else None}


class BaseTrainer:
    """
    Base class for iterative trainer with hooks.
    """

    def __init__(self, config: DictConfig) -> None:
        self.cfg = config
        # Pristine deep copy captured BEFORE any component construction mutates
        # the tree (e.g. Inferencer merges model output metadata into it).
        # Checkpoint metadata hashes/persists THIS copy so the recorded config
        # reflects what the run was launched with.
        self.pristine_cfg = copy.deepcopy(config)
        # initialize event recorder
        self.event_recorder: EventRecorder = instantiate(_ensure_full_path(config.loggers.event_recorder))

        # initialize event_writers
        event_writers = self._build_event_writers(w_cfgs=config.loggers.event_writers, recorder=self.event_recorder)
        self.event_writers_list = instantiate(
            _ensure_full_path(config.loggers.event_writers_list), writers=event_writers
        )
        for writer in event_writers:
            save_config = getattr(writer, "save_config", None)
            if save_config is not None:
                save_config(config)

        # intialize hooks
        hooks = self._build_hooks(config.hooks.hooks_list, self.event_writers_list)
        self._hooks: List[HookBase] = []
        self.register_hooks(hooks)

    @staticmethod
    def _build_event_writers(w_cfgs, recorder):
        writers = []
        for writer_cfg in w_cfgs:
            writer = REGISTRY.build(
                "event_writer", writer_cfg.name, writer_cfg, event_recorder=recorder
            )
            writers.append(writer)
        return writers

    @staticmethod
    def _build_hooks(h_cfgs, event_writers):
        hooks = []
        for hc in h_cfgs:
            # inject writers into the PeriodicWriter hook
            overrides = {"writers": event_writers} if hc.name == "periodic_writer" else {}
            hooks.append(REGISTRY.build("hook", hc.name, hc, **overrides))
        return hooks

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        """
        Register hooks to the trainer. The hooks are executed
        in the order they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]

        for h in hooks:
            # isinstance (not a direct-subclass name check): grand-children of
            # HookBase and same-name classes from other modules were previously
            # mis-handled.
            assert isinstance(h, HookBase), (
                f"Hook {h!r} must be a HookBase instance, got {type(h)}"
            )
            # to avoid circular reference, hooks and trainer
            # cannot own each other this normally does not
            # matter, but will cause memory leak if the
            # involved objects contain __del__
            # hence we use weakref.proxy
            h.trainer = weakref.proxy(self)
        
        # reorder hooks by priority
        # higher priority hooks are executed first
        hooks.sort(key=lambda h: -h.PRIORITY.value)
        self._hooks.extend(hooks)

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        self.event_recorder._iter = self._iter
        self.event_recorder._epoch = self._epoch
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        # maintain the invariant that 
        # event_recorder.iter == trainer.iter
        # for the entire execution of each step
        self.event_recorder._iter = self._iter
        for h in self._hooks:
            h.before_step()

    def after_backward(self, data_sample: Any, 
                       loss_dict: Dict[str, Any], 
                       outputs: Optional[Any] = None
    ):
        for h in self._hooks:
            h.after_backward(data_sample=data_sample, 
                             loss_dict=loss_dict, 
                             outputs=outputs)

    def before_backward(
        self,
        data_sample: Any,
        loss_dict: Dict[str, Any],
        outputs: Optional[Any] = None,
    ):
        for h in self._hooks:
            h.before_backward(
                data_sample=data_sample,
                loss_dict=loss_dict,
                outputs=outputs,
            )

    def after_step(self,*args, **kwargs):
        for h in self._hooks:
            h.after_step(*args, **kwargs)

    def before_epoch(self):
        # maintain the invariant that 
        # event_recorder.epoch == trainer.epoch
        # for the entire execution of each step
        self.event_recorder._epoch = self._epoch
        for h in self._hooks:
            h.before_epoch()
    
    def after_epoch(self, *args, **kwargs):
        for h in self._hooks:
            h.after_epoch(*args, **kwargs)

    def before_validation(self):
        self.event_recorder._val_iter = 0
        for h in self._hooks:
            h.before_validation()

    def after_validation(self):
        for h in self._hooks:
            h.after_validation()
    
    def before_val_step(self):
        # maintain the invariant that 
        # event_recorder.val_iter == trainer.val_iter
        # for the entire execution of each validation step
        self.event_recorder._val_iter = self._val_iter
        for h in self._hooks:
            h.before_val_step()

    def after_val_step(self, *args, **kwargs):
        for h in self._hooks:
            h.after_val_step(*args, **kwargs)

    def before_test(self):
        self.event_recorder._iter = 0
        for h in self._hooks:
            h.before_test()

    def after_test(self):
        self.event_recorder._iter = self._iter
        for h in self._hooks:
            h.after_test()
    
    def before_test_step(self):
        # maintain the invariant that 
        # event_recorder.test_iter == trainer.test_iter
        # for the entire execution of each test step
        self.event_recorder._iter = self._iter
        for h in self._hooks:
            h.before_test_step()
    
    def after_test_step(self, *args, **kwargs):
        for h in self._hooks:
            h.after_test_step(*args, **kwargs)

    def state_dict(self):
        ret = {"iteration": self._iter, "epoch": self._epoch}

        if hasattr(self, "best_metric"):
            ret["best_metric"] = self.best_metric
        if hasattr(self, "best_metric_epoch"):
            ret["best_metric_epoch"] = self.best_metric_epoch
        if hasattr(self, "best_metric_iter"):
            ret["best_metric_iter"] = self.best_metric_iter

        hooks_state = {}
        for h in self._hooks:
            sd = h.state_dict()
            if sd:
                name = type(h).__qualname__
                if name in hooks_state:
                    # TODO: handle repetitive stateful hooks
                    continue
                hooks_state[name] = sd
        if hooks_state:
            ret["hooks"] = hooks_state
        return ret

    def load_state_dict(self, state_dict):
        self._iter = state_dict["iteration"]
        self._epoch = state_dict["epoch"]

        if "best_metric" in state_dict:
            self.best_metric = state_dict["best_metric"]
        if "best_metric_epoch" in state_dict:
            self.best_metric_epoch = state_dict["best_metric_epoch"]
        if "best_metric_iter" in state_dict:
            self.best_metric_iter = state_dict["best_metric_iter"]

        for key, value in state_dict.get("hooks", {}).items():
            for h in self._hooks:
                try:
                    name = type(h).__qualname__
                except AttributeError:
                    continue
                if name == key:
                    h.load_state_dict(value)
                    break
            else:
                logger.warning(f"Cannot find the hook '{key}', its state_dict is ignored.")


class ValidationLoopMixin:
    """In-loop validation shared by EpochBasedTrainer and TorchNativeTrainer.

    Requires: ``model``, ``preprocessor``, ``evaluator``, ``event_recorder``,
    ``_val_iter``, and the BaseTrainer hook dispatch surface.
    """

    def run_validation(self, val_dataloader) -> None:
        """
        Run validation.
        """
        # In-loop validation runs the training forward pass
        # (`self.model(data_sample)` -> loss + outputs) and feeds those outputs to
        # a loss-based evaluator (e.g. BaseEvaluator). Prediction-based evaluators
        # consume `model.evaluate_step` outputs and belong to job_type=test.
        self.before_validation()
        # technically, contexts could be a hook
        # but kept here for clarity
        with inference_context(self.model):
            with torch.no_grad():
                end = time.perf_counter()
                for idx, data_sample in enumerate(val_dataloader):
                    data_time = time.perf_counter() - end
                    data_sample = self.preprocessor(data_sample=data_sample, data_time=data_time, idx=idx)
                    # run one step with the fetched data sample
                    self.run_validation_step(idx, data_sample)
                    end = time.perf_counter()

        # Loss is the model's (logged per step by log_loss_dict); the evaluator
        # only adds prediction metrics on top. Drop any evaluator output already
        # logged as loss this epoch so BaseEvaluator's step_loss doesn't
        # double-source and skew the reduction.
        metrics = self.evaluator.evaluate()
        already_logged = set(self.event_recorder.get_epoch_scalars().keys())
        evaluator_metrics = {}
        for k, v in metrics.items():
            key = get_metric_full_name(name=k, scope="epoch", category="loss", prefix="val")
            if key in already_logged:
                continue
            evaluator_metrics[k] = v.item() if torch.is_tensor(v) else v
        if evaluator_metrics:
            self.event_recorder.put_scalars(
                scope="epoch",
                prefix="val",
                category="loss",
                **evaluator_metrics,
            )
        self.evaluator.reset()

        self.after_validation()

    def run_validation_step(self, idx: int, data_sample: Sequence[dict]) -> None:
        """
        Iterate one validation step.
        """
        self.before_val_step()

        loss_dict, outputs = self.model(data_sample)
        # detached GPU scalars; materialized at the logging boundary
        # (log_loss_dict) — see run_step
        loss_dict_log = {
            k: (v.detach() if torch.is_tensor(v) else v)
            for k, v in loss_dict.items()
        }
        self.evaluator.process(data_sample, outputs, loss_dict)

        self.after_val_step(
            data_sample={"metainfo": data_sample.get("metainfo")},
            outputs=None,
            loss_dict=loss_dict_log
        )

        outputs = None
        loss_dict = None
        loss_dict_log = None
        data_sample = None

        self._val_iter += 1


class EpochBasedTrainer(ValidationLoopMixin, BaseTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        # Seed every RNG (python/numpy/torch/cuda) before ANY component that
        # draws randomness is built (model init, dataloader, transforms).
        # Rank-decorrelated so per-rank augmentation noise differs while runs
        # stay reproducible.
        # NOTE: rank-decorrelated seeding is safe HERE ONLY because
        # deepspeed.initialize broadcasts rank0 weights before training.
        # TorchNativeTrainer must build the model with an IDENTICAL seed on all
        # ranks (fully_shard slices each rank's local copy in place) and only
        # re-seeds with +process_rank() after the model exists. Keep in sync.
        _seed = int(cfg.seed) + process_rank()
        set_global_seed(_seed)
        logger.info(f"[Trainer] seeded rank {process_rank()} with {_seed} (cfg.seed={cfg.seed})")

        print(f"Run Config:\n{OmegaConf.to_yaml(cfg)}")

        self.ray_context = get_context()
        self.val_begin, self.val_interval = cfg.evaluation.val_begin, cfg.evaluation.val_interval
        self.stop_training, self._max_epochs = False, cfg.schedulers.epochs

        dp_degree = get_world_size()
        # Verify batch sizes
        if cfg.clusters.batch_size % (cfg.clusters.batch_size_per_gpu * dp_degree) != 0:
            raise ValueError(
                f"global batch size must be multiple of local batch size times "
                f"data-parallel degree ({cfg.clusters.batch_size} "
                f"% ({cfg.clusters.batch_size_per_gpu} * {dp_degree}) != 0)"
            )
        # Calculate gradient accumulation steps
        self.gradient_accumulation_steps = cfg.clusters.batch_size // (
            cfg.clusters.batch_size_per_gpu * dp_degree
        )
        if cfg.deepspeed.gradient_accumulation_steps != self.gradient_accumulation_steps:
            raise ValueError(
                "Deepspeed gradient_accumulation_steps does not match the calculated value."
            )
        self.with_grad_accumulation = self.gradient_accumulation_steps > 1
        if self.gradient_accumulation_steps != 1:
            # Hard raise, NOT an assert: under python -O a stripped assert
            # silently enabled accumulation, whose LR/WD hooks step per
            # micro-batch (schedule compression) -- the boundary gating now
            # exists (run_step._at_accumulation_boundary) but is unvalidated.
            raise NotImplementedError(
                "Gradient accumulation is not supported yet: validate the "
                "LR/WD boundary gating and FreeDeviceBufferHook's "
                "after_backward path before enabling batch_size > "
                "batch_size_per_gpu * world_size."
            )

        # initialize dataset and dataloader
        _, _, self.dataloader_config, self.host_buffer_actor, self.device_buffer, _ = get_dataloader(cfg)

        self.steps_per_epoch, self.val_steps_per_epoch = get_steps_per_epoch(
            config=cfg,
            gradient_accumulation_steps=self.gradient_accumulation_steps
        )

        # initialize model
        model = REGISTRY.build("model", cfg.models.model, cfg)

        # NOTE: we are moving model weight initialization back into each model class
        #       this may be problematic as we scale to larger models. However, 
        #       currently it exposes the possibility of bugs to separate model init 
        #       from model build.
        # with torch.no_grad():
        #     model.init_model_weights(buffer_device="cuda")

        # FIXME: temporarily disable flops and param counting
        # if hasattr(model, "_get_nparams_and_flops"):
        #     # NOTE: used to calculate flops and params for logging purposes
        #     self.model_param_count, self.num_flops_per_token = \
        #         model._get_nparams_and_flops(batch_size=cfg.clusters.batch_size_per_gpu, device="meta")
        # else:
        #     logger.warning(
        #         "Model does not implement `_get_nparams_and_flops` method. "
        #         "Setting model_param_count and num_flops_per_token to -1."
        #         "Flops and parameter counts will be unavailable in reported metrics."
        #     )
        #     self.model_param_count, self.num_flops_per_token = -1, -1

        self.preprocessor = REGISTRY.build("preprocessor", cfg.datasets.preprocessor.name, cfg.datasets.preprocessor)

        # FIXME: not always desirable to force load model weights from this checkpoint
        # initialize checkpoint manager
        # if os.environ.get("RESTART", "FALSE").upper() == "TRUE":
        #     logger.info("RESTART flag detected. Resuming from latest checkpoint.")
        #     with open_dict(cfg):
        #         cfg.paths.resume_checkpointdir = Path(cfg.paths.outdir) / "checkpoints"
        #         cfg.checkpoint.checkpoint_manager.resume_checkpointdir = cfg.paths.resume_checkpointdir

        self.checkpoint_manager = instantiate(
            cfg.checkpoint.checkpoint_manager,
            model=model
        )

        if cfg.checkpoint.checkpoint_manager.pretrained_checkpointdir:
            _, _ = self.checkpoint_manager.load()

        if cfg.optimizations.with_model_summary:
            rank = process_rank()
            if rank == 0:
                # FIXME: make the get_masked_input_data function compatible with all models.
                #        Consider adding helper to each model class to generate input data.
                try:
                    # Per-GPU batch: the summary runs on ONE rank; the global
                    # batch size would inflate activation-memory estimates
                    # (and can OOM the summary itself) by the DP degree.
                    input_shape = (cfg.clusters.batch_size_per_gpu, *cfg.datasets.input_shape)
                    input_data = get_masked_input_data(model, input_shape)
        
                    summarize_model(
                        model=model,
                        inputs=input_shape,
                        input_data=input_data,
                        # per-GPU batch: the byte/MAC figures in the same logbook
                        # are computed at batch_size_per_gpu (the summary forward)
                        batch_size=cfg.clusters.batch_size_per_gpu,
                        logdir=cfg.paths.outdir,
                    )
                except Exception as e:
                    logger.warning(
                        f"Model summary skipped (likely incompatible model/input_data): {e}"
                    )

        # initialize optimizer and learning rate scheduler
        param_groups = get_param_groups(cfg, model)
        self.opt, _ = get_optimizer(
            params=param_groups,
            config=cfg,
            optimizer=cfg.optimizers.name,
            steps_per_epoch=self.steps_per_epoch
        )
        self.schedulers, self.wd_schedulers = get_schedulers(
            opt=self.opt,
            config=cfg,
            steps_per_epoch=self.steps_per_epoch
        )

        # enable optimizations if specified
        # includes setting Torch backend flags, 
        # activation checkpointing, and torch Compile 
        if cfg.optimizations is not None:
            enable_optimizations(cfg=cfg)
        opt_cfg = get_model_optimizations_node(cfg)
        if opt_cfg.activation_checkpoint.enable:
            logger.info("[Trainer] Applying activation checkpointing...")
            apply_activation_checkpointing(opt_cfg, model)
        if opt_cfg.torch_compile.enable:
            logger.info("[Trainer] Applying torch.compile...")
            model = apply_compile(opt_cfg, model)

        # initialize deepspeed (imported lazily — see module-level note)
        from deepspeed import initialize as _deepspeed_initialize

        self.model, self.optimizers, _, _ = _deepspeed_initialize(
            model=model,
            optimizer=self.opt,
            config=OmegaConf.to_container(cfg.deepspeed, resolve=True)
        )
        self.checkpoint_manager.model = self.model

        # if resume job, gather the state from the checkpoint
        # else intialize outdir, logdir, and checkpointdir
        # these directories should be empty if not resuming a job
        # to avoid overwriting existing checkpoints
        best_metric, step, epoch = resume_run(self, cfg)
        self.start_epoch, self.start_iter, self.best_metric = epoch, step, best_metric
        # _curr_val_metric is "latest validation metric"; initialize with the
        # mode-aware sentinel (+inf for min, -inf for max) so max-mode consumers
        # never see a +inf that reads as an unbeatable best.
        self._epoch, self._iter, self._val_iter = self.start_epoch, self.start_iter, 0
        self._curr_val_metric = initial_best_metric(cfg)

        if self.start_iter > 0:
            logger.info("[Trainer] Resuming training; DeepSpeed restores optimizer state (load_optimizer_states=True).")
            logger.info(f"[Trainer] Fast forwarding lr and wd schedulers to iter {self.start_iter} and epoch {self.start_epoch}.")
            # fast forward lr and wd schedulers to the correct step
            # TODO: consider making more flexible
            if self.wd_schedulers is not None:
                for _ in range(self.start_iter):
                    self.wd_schedulers.step()
            
            if self.schedulers.update_type == "epoch":
                # Mirror the live cadence (LRScheduler.after_epoch steps with
                # epoch+1 after each completed epoch): one call per completed
                # epoch, values 1..start_epoch. Count-based schedulers see the
                # right number of steps; value-based ones land on LR(start_epoch).
                for epoch in range(1, self.start_epoch + 1):
                    self.schedulers.step(epoch)
            elif self.schedulers.update_type == "step":
                for iter in range(self.start_iter):
                    self.schedulers.step(iter)
            else:
                raise NotImplementedError(f'{self.schedulers.update_type=} is not supported')

            # One-shot re-apply after the replay: guarantees the optimizer's
            # param-group LRs match the scheduler's replayed position before
            # the first resumed optimizer step. TODO: consider removing.
            if hasattr(self.schedulers, "_apply") and hasattr(self.schedulers, "_step"):
                self.schedulers._apply(self.schedulers._step)

        # initialize evaluator
        self.evaluator = REGISTRY.build("evaluator", cfg.evaluation.evaluator.name, cfg.evaluation.evaluator)

    def run(self):
        """
        Launch training.
        """
        self.before_train()

        while self._epoch < self._max_epochs and not self.stop_training:
            self.run_epoch()

        self.after_train()

    def run_epoch(self) -> None:
        """
        Iterate one epoch.
        """
        self.before_epoch()

        train_dataloader, val_dataloader, _ = get_dataloader_ray(
            **self.dataloader_config,
            epoch=self._epoch
        )

        observed_steps = 0
        end = time.perf_counter()
        for idx, data_sample in enumerate(train_dataloader):
            data_time = time.perf_counter() - end
            data_sample = self.preprocessor(data_sample=data_sample, data_time=data_time, idx=idx)
            # run one step with the fetched data sample
            self.run_step(idx, data_sample)
            observed_steps = idx + 1
            end = time.perf_counter()

        # LR/WD schedules size their T_max from steps_per_epoch; a dataloader
        # that yields a different count silently desyncs every schedule. Check
        # once, after the first completed epoch of this run.
        if not getattr(self, "_steps_per_epoch_validated", False):
            # TODO: could this change epoch-to-epoch?
            if observed_steps != self.steps_per_epoch:
                raise RuntimeError(
                    f"Observed {observed_steps} train steps in epoch {self._epoch} but "
                    f"steps_per_epoch={self.steps_per_epoch}. LR/WD scheduler horizons "
                    f"(T_max) are computed from steps_per_epoch and would desync — fix "
                    f"the steps-per-epoch inference or the dataloader batch policy."
                )
            self._steps_per_epoch_validated = True

        if val_dataloader and (
            self._epoch >= self.val_begin and (self._epoch - self.val_begin) % self.val_interval == 0
        ):
            # run validation
            self.run_validation(val_dataloader)

        self.after_epoch()
        self._epoch += 1

    def run_step(self, idx, data_sample: Sequence[dict]) -> None:
        """
        Iterate one mini-batch.
        """
        self.before_step()

        # we enforce that all models compute
        # their losses in the forward pass
        # and return a loss_dict with at least
        # a "step_loss" key together with
        # the outputs of the model
        meta = data_sample.get("metainfo", None)
        if meta is None:
            # Fail here (with context) instead of inside FreeDeviceBufferHook,
            # which would otherwise crash on None["metainfo"] after leaking the
            # device-buffer slot.
            raise RuntimeError(
                "data_sample['metainfo'] is None after preprocessing — hooks "
                "(FreeDeviceBufferHook, metric logging) require metainfo; the "
                "preprocessor must always emit it."
            )
        loss_dict, outputs = self.model(data_sample)

        del outputs  # Need to free outputs before bkwd to avoid peaking memory
        loss = loss_dict["step_loss"]
        # Keep DETACHED GPU scalars — no .cpu()/.item() here: that forced a
        # device sync before backward every step. The values materialize once
        # per logging boundary (log_loss_dict / hooks call .item() after
        # model.step()).
        loss_dict_log = {
            k: (v.detach() if torch.is_tensor(v) else v)
            for k, v in loss_dict.items()
        }

        self.before_backward(data_sample=data_sample, loss_dict=loss_dict_log, outputs=None)

        outputs = None
        data_sample = None
        loss_dict = None

        self.model.backward(loss)

        # after_backward phase: grads exist, optimizer has NOT stepped.
        # FreeDeviceBufferHook returns device-buffer slots here under grad
        # accumulation (the input is no longer needed once grads are formed);
        # this phase previously had NO dispatch site in the live trainer.
        self.after_backward(
            data_sample={"metainfo": meta},
            outputs=None,
            loss_dict=loss_dict_log,
        )

        # Capture the accumulation boundary BEFORE model.step(): DeepSpeed
        # advances micro_steps inside step(), so querying afterwards refers to
        # the NEXT micro-batch (off-by-one for LR/WD gating when accumulation
        # is enabled).
        # Exposed as trainer state (not an after_step kwarg) so hook signatures
        # stay untouched; LR/WD hooks read it via
        # getattr(trainer, "_at_accumulation_boundary", True).
        self._at_accumulation_boundary = (
            not hasattr(self.model, "is_gradient_accumulation_boundary")
            or self.model.is_gradient_accumulation_boundary()
        )
        self.model.step()

        # for short testing runs:
        # if idx > 25:
        #     raise RuntimeError(
        #         f"Training stopped at step {idx} for testing."
        #     )
        # logger.info(f"step_loss: {loss_dict['step_loss']}, lr: {self.opt.param_groups[0]['lr']}")

        self.after_step(
            data_sample={"metainfo": meta},
            outputs=None,
            loss_dict=loss_dict_log,
        )
        self._iter += 1

    # run_validation / run_validation_step: provided by ValidationLoopMixin.


def _init_cuda_event_timers() -> dict:
    """fwd/bwd/step cuda-event pairs consumed by loggers.MetricsProcessor."""
    return {
        name: torch.cuda.Event(enable_timing=True)
        for name in (
            "fwd_start", "fwd_end",
            "bwd_start", "bwd_end",
            "step_start", "step_end",
        )
    }


class TorchNativeTrainer(ValidationLoopMixin, BaseTrainer):
    """Torch-native trainer: FSDP2 (fully_shard) + optional FP8/MX linears +
    per-block torch.compile. Supports step- and epoch-based iteration.

    Parallelism is pure data parallelism: FSDP over ``dp_shard`` ranks,
    optionally HSDP with ``dp_replicate > 1``. No TP/CP/PP/EP.

    Gradient accumulation is NOT supported (rejected in ``__init__``, parity with
    EpochBasedTrainer): one batch == one optimizer step.

    Construction order is load-bearing:
      seed(identical) -> mesh -> dataloader -> model build (on device) ->
      quantize -> AC -> compile -> fully_shard -> optimizer -> schedulers ->
      DCP checkpointer -> resume.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        if cfg.backend.upper() != "TORCHTITAN":
            raise ValueError(
                f"TorchNativeTrainer requires backend: TORCHTITAN (got {cfg.backend!r}) "
                "so the backend-branching hooks take their torch-native paths."
            )

        # Seed every RNG with an IDENTICAL seed on all ranks: the model is
        # materialized on-device and fully_shard slices each rank's local copy
        # in place, so divergent init would silently produce an inconsistent
        # global model. (DeepSpeed hides this by broadcasting rank0 weights at
        # engine init; FSDP2 does not.) Rank-decorrelated re-seed happens after
        # the model exists, below.
        set_global_seed(int(cfg.seed))
        logger.info(f"[Trainer] seeded rank {process_rank()} with {int(cfg.seed)} (identical across ranks for model init)")

        if is_main_process():
            print(f"Run Config:\n{OmegaConf.to_yaml(cfg)}")

        self.ray_context = get_context()
        self.stop_training = False
        self._at_accumulation_boundary = True
        # placeholder counters: the DCP checkpointer snapshots trainer state
        # via BaseTrainer.state_dict() during resume, before the real values
        # below are known
        self._epoch, self._iter, self._val_iter = 0, 0, 0

        loop_cfg = cfg.trainer_loop
        self.iteration_mode = str(loop_cfg.iteration_mode)
        if self.iteration_mode not in ("epoch", "step"):
            raise ValueError(
                f"trainer_loop.iteration_mode must be 'epoch' or 'step', got {self.iteration_mode!r}"
            )

        # NCCL comm env (flight recorder, async error handling) + torch backend flags
        configure_torch_comm_env(cfg.parallelism.comm)
        if cfg.optimizations is not None:
            enable_optimizations(cfg=cfg)

        # ------------------------------------------------------------------ #
        # Device mesh. Ray Train's TorchTrainer already ran init_process_group;
        # we only build the mesh. dp_replicate > 1 => HSDP (2D mesh: replicate
        # across dim 0, shard within dim 1); else pure FSDP over all ranks.
        # ------------------------------------------------------------------ #
        world_size = get_world_size()
        t_cfg = cfg.parallelism.training
        dp_replicate = int(t_cfg.data_parallel_replicate_degree)
        dp_shard = int(t_cfg.data_parallel_shard_degree)
        if dp_shard == -1:
            dp_shard = world_size // dp_replicate
        if dp_replicate * dp_shard != world_size:
            raise ValueError(
                f"dp_replicate * dp_shard must equal world_size "
                f"({dp_replicate} * {dp_shard} != {world_size})"
            )
        self.max_norm = float(t_cfg.max_norm)

        self.device = torch.device("cuda", local_rank())
        torch.cuda.set_device(self.device)

        from torch.distributed.device_mesh import init_device_mesh

        if dp_replicate > 1:
            self.dp_mesh = init_device_mesh(
                "cuda", (dp_replicate, dp_shard),
                mesh_dim_names=("dp_replicate", "dp_shard"),
            )
        else:
            self.dp_mesh = init_device_mesh(
                "cuda", (dp_shard,), mesh_dim_names=("dp_shard",)
            )
        # every rank consumes distinct data regardless of the replicate/shard split
        self.dp_degree, self.dp_rank = world_size, process_rank()

        # bookkeeping object only (MetricsProcessor reads non_data_parallel_size);
        # the mesh above is authoritative — ParallelDims 0.2.0 builds no named
        # mesh dims at degree 1, so its world_mesh is NOT used here.
        from torchtitan.distributed import ParallelDims

        self.parallel_dims = ParallelDims(
            dp_replicate=dp_replicate, dp_shard=dp_shard,
            cp=1, tp=1, pp=1, ep=1, etp=1, world_size=world_size,
        )

        # ------------------------------------------------------------------ #
        # Batch accounting (same invariant as EpochBasedTrainer; gradient
        # accumulation is REJECTED below, so one batch == one optimizer step and
        # every scheduler/hook keys off optimizer steps)
        # ------------------------------------------------------------------ #
        if cfg.clusters.batch_size % (cfg.clusters.batch_size_per_gpu * world_size) != 0:
            raise ValueError(
                f"global batch size must be multiple of local batch size times "
                f"data-parallel degree ({cfg.clusters.batch_size} "
                f"% ({cfg.clusters.batch_size_per_gpu} * {world_size}) != 0)"
            )
        self.gradient_accumulation_steps = cfg.clusters.batch_size // (
            cfg.clusters.batch_size_per_gpu * world_size
        )
        # Hard raise, NOT an assert: under python -O a stripped assert silently
        # enables accumulation. Parity with EpochBasedTrainer -- the LR/WD
        # boundary gating (_at_accumulation_boundary) and FreeDeviceBufferHook's
        # after_backward path exist but are unvalidated on the FSDP2 path, and
        # the trailing-partial-window drop only executes on a ragged epoch
        # boundary (so it would first run in production).
        if self.gradient_accumulation_steps != 1:
            raise NotImplementedError(
                "Gradient accumulation is not supported by TorchNativeTrainer: got "
                f"gradient_accumulation_steps={self.gradient_accumulation_steps} from "
                f"batch_size={cfg.clusters.batch_size} / "
                f"(batch_size_per_gpu={cfg.clusters.batch_size_per_gpu} * "
                f"world_size={world_size}). Set batch_size == batch_size_per_gpu * "
                "world_size. Re-enabling requires validating the LR/WD boundary "
                "gating and FreeDeviceBufferHook.after_backward first."
            )
        # Kept (== 1) so get_steps_per_epoch keeps its contract and
        # FreeDeviceBufferHook keeps freeing in after_step (it reads this flag in
        # before_train); re-enabling later is then a small diff.
        self.with_grad_accumulation = False

        # ------------------------------------------------------------------ #
        # Dataloader + step horizons
        # ------------------------------------------------------------------ #
        _, _, self.dataloader_config, self.host_buffer_actor, self.device_buffer, _ = get_dataloader(
            cfg, dp_degree=self.dp_degree, dp_rank=self.dp_rank
        )
        self.steps_per_epoch, self.val_steps_per_epoch = get_steps_per_epoch(
            config=cfg,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )

        if self.iteration_mode == "epoch":
            self._max_epochs = int(cfg.schedulers.epochs)
            self.total_steps = self.steps_per_epoch * self._max_epochs
        else:
            max_steps = loop_cfg.get("max_steps", None)
            if not max_steps:
                raise ValueError("trainer_loop.iteration_mode=step requires trainer_loop.max_steps")
            self.total_steps = int(max_steps)
            self._max_epochs = math.ceil(self.total_steps / self.steps_per_epoch)
        self.val_begin, self.val_interval = cfg.evaluation.val_begin, cfg.evaluation.val_interval
        self.val_every_n_steps = loop_cfg.get("val_every_n_steps", None)

        # checkpoint cadence in OPTIMIZER STEPS; epoch-mode convenience knob
        # `checkpoint_every_n_epochs` resolves to aligned step counts (epoch
        # boundaries land exactly on multiples of steps_per_epoch)
        every_n_epochs = loop_cfg.get("checkpoint_every_n_epochs", None)
        if every_n_epochs:
            self.checkpoint_save_period = int(every_n_epochs) * self.steps_per_epoch
        else:
            self.checkpoint_save_period = int(
                loop_cfg.get("checkpoint_save_period", None) or self.steps_per_epoch
            )

        # ------------------------------------------------------------------ #
        # Model: build directly on device (identical weights on every rank —
        # see seed note above), then quantize -> AC -> compile -> fully_shard.
        # Meta-device init is not implemented yet.
        # ------------------------------------------------------------------ #
        with torch.device(self.device):
            model = REGISTRY.build("model", cfg.models.model, cfg)

        self.preprocessor = REGISTRY.build(
            "preprocessor", cfg.datasets.preprocessor.name, cfg.datasets.preprocessor
        )

        if cfg.optimizations.with_model_summary and is_main_process():
            try:
                input_shape = (cfg.clusters.batch_size_per_gpu, *cfg.datasets.input_shape)
                input_data = get_masked_input_data(model, input_shape)
                summarize_model(
                    model=model,
                    inputs=input_shape,
                    input_data=input_data,
                    batch_size=cfg.clusters.batch_size_per_gpu,
                    logdir=cfg.paths.outdir,
                )
            except Exception as e:
                logger.warning(
                    f"Model summary skipped (likely incompatible model/input_data): {e}"
                )

        model, self.quantize_converter = parallelize(model, self.dp_mesh, cfg)
        model.train()
        self.model = model
        self.model_parts = [model]

        # weights exist and are sharded — decorrelate per-rank randomness for
        # data augmentation / dropout from here on
        set_global_seed(int(cfg.seed) + process_rank())

        # ------------------------------------------------------------------ #
        # Optimizer + LR/WD schedulers (containers are Stateful: DCP restores
        # them directly, no fast-forward replay needed on resume)
        # ------------------------------------------------------------------ #
        param_groups = get_param_groups(cfg, model)
        self.optimizers = build_optimizers(
            model_parts=self.model_parts,
            optimizer_config=cfg.optimizers,
            param_groups=param_groups,
        )
        self.schedulers = build_lr_schedulers(
            self.optimizers,
            cfg.schedulers,
            training_steps=self.total_steps,
            steps_per_epoch=self.steps_per_epoch,
        )
        self.wd_schedulers = build_wd_schedulers(
            self.optimizers,
            cfg.schedulers.get("wd_scheduler", {}),
            training_steps=self.total_steps,
        )

        if self.quantize_converter is not None:
            # e.g. tensorwise-FP8 dynamic amax/scale precompute for the FSDP2
            # fp8 all-gather (no-op for other recipes)
            self.optimizers.register_step_post_hook(
                lambda *args, **kwargs: self.quantize_converter.post_optimizer_hook(
                    self.model_parts
                )
            )

        # ------------------------------------------------------------------ #
        # Perf metrics (tps/tflops/MFU + fwd/bwd/step timings) — optional
        # ------------------------------------------------------------------ #
        if loop_cfg.get("with_perf_metrics", False):
            self.timers = _init_cuda_event_timers()
            self.metrics_processor = MetricsProcessor(
                timers=self.timers,
                parallel_dims=self.parallel_dims,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                optimizers=self.optimizers,
                lr_schedulers=self.schedulers,
                model_parts=self.model_parts,
            )
        else:
            self.timers, self.metrics_processor = None, None

        # ------------------------------------------------------------------ #
        # DCP checkpointing + resume
        # ------------------------------------------------------------------ #
        extra_states = {"train_state": self}
        if self.wd_schedulers is not None:
            extra_states["wd_schedulers"] = self.wd_schedulers
        self.checkpoint_manager = instantiate(
            cfg.checkpoint.checkpoint_manager,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.schedulers,
            states=extra_states,
            save_period=self.checkpoint_save_period,
        )

        if cfg.checkpoint.checkpoint_manager.pretrained_checkpointdir:
            # warm start: model weights only
            self.checkpoint_manager.load()

        best_metric, step, epoch = resume_run(self, cfg)
        self.start_epoch, self.start_iter, self.best_metric = epoch, step, best_metric
        self._iter, self._val_iter = self.start_iter, 0
        # Derive epoch + in-epoch offset from the global step count: this
        # self-heals the save-at-epoch-boundary case (where the persisted
        # _epoch lags the boundary by one) and drives mid-epoch dataloader
        # skip on resume (run_epoch).
        self._epoch = self._iter // self.steps_per_epoch
        self._epoch_step_offset = self._iter % self.steps_per_epoch
        if self._epoch != epoch:
            logger.info(
                f"[Trainer] normalized resume epoch {epoch} -> {self._epoch} "
                f"(iter={self._iter}, steps_per_epoch={self.steps_per_epoch})"
            )
        self.start_epoch = self._epoch
        self._curr_val_metric = initial_best_metric(cfg)
        self._pg_timeouts_adjusted = False

        # initialize evaluator
        self.evaluator = REGISTRY.build(
            "evaluator", cfg.evaluation.evaluator.name, cfg.evaluation.evaluator
        )

    # -------------------------------------------------------------- loop --
    def _done(self) -> bool:
        if self.stop_training or self._iter >= self.total_steps:
            return True
        return self.iteration_mode == "epoch" and self._epoch >= self._max_epochs

    def run(self):
        """
        Launch training.
        """
        self.before_train()

        while not self._done():
            self.run_epoch()

        self.after_train()

    def run_epoch(self) -> None:
        """
        Iterate one epoch: one batch per optimizer step (no gradient
        accumulation, see __init__) until the dataloader is exhausted or the
        step budget is hit (step mode).
        """
        self.before_epoch()

        # mid-epoch resume: skip the already-consumed batches of THIS epoch;
        # the shuffle is deterministic in (seed, epoch), so the remaining
        # order matches the interrupted run. One batch per optimizer step, so
        # the batch offset IS the step offset.
        skip_batches = 0
        if self._epoch == self.start_epoch and self._epoch_step_offset > 0:
            skip_batches = self._epoch_step_offset
            logger.info(
                f"[Trainer] mid-epoch resume: skipping {skip_batches} batches "
                f"of epoch {self._epoch}"
            )

        train_dataloader, val_dataloader, _ = get_dataloader_ray(
            **self.dataloader_config,
            epoch=self._epoch,
            skip_batches=skip_batches,
        )
        data_iter = iter(train_dataloader)

        observed_steps, exhausted = 0, False
        end = time.perf_counter()
        while not self._done():
            try:
                data_sample = next(data_iter)
            except StopIteration:
                # No accumulation window to unwind: nothing has been
                # preprocessed yet this iteration, so no device-buffer slot is
                # outstanding and there is nothing to return to the pool.
                exhausted = True
                break

            data_time = time.perf_counter() - end
            microbatch = self.preprocessor(
                data_sample=data_sample, data_time=data_time, idx=self._iter
            )
            end = time.perf_counter()

            self.run_step(microbatch)
            observed_steps += 1

            # reduce the (long) init PG timeout once real training is flowing
            if not self._pg_timeouts_adjusted and self._iter == self.start_iter + 1:
                from torchtitan.distributed import utils as dist_utils  # lazy

                dist_utils.set_pg_timeouts(
                    timeout=timedelta(
                        seconds=self.cfg.parallelism.comm.train_timeout_seconds
                    ),
                    world_mesh=self.dp_mesh,
                )
                self._pg_timeouts_adjusted = True

            # step-cadence validation
            if (
                val_dataloader
                and self.val_every_n_steps
                and self._iter % self.val_every_n_steps == 0
            ):
                self.run_validation(val_dataloader)

        # LR/WD schedules size their horizons from steps_per_epoch; validate the
        # first FULLY OBSERVED epoch of this run against the inferred count.
        if exhausted and not getattr(self, "_steps_per_epoch_validated", False):
            expected = self.steps_per_epoch - skip_batches
            if observed_steps != expected:
                raise RuntimeError(
                    f"Observed {observed_steps} train steps in epoch {self._epoch} but "
                    f"expected {expected} (steps_per_epoch={self.steps_per_epoch}, "
                    f"skipped={skip_batches}). "
                    f"LR/WD scheduler horizons are computed from steps_per_epoch and "
                    f"would desync — fix the steps-per-epoch inference or the "
                    f"dataloader batch policy."
                )
            self._steps_per_epoch_validated = True

        # epoch-cadence validation (used when val_every_n_steps is unset)
        if (
            val_dataloader
            and not self.val_every_n_steps
            and self._epoch >= self.val_begin
            and (self._epoch - self.val_begin) % self.val_interval == 0
        ):
            self.run_validation(val_dataloader)

        self.after_epoch()
        self._epoch += 1

    def run_step(self, data_sample: dict) -> None:
        """
        One optimizer step over ONE preprocessed batch: zero_grad -> forward ->
        backward -> clip_grad_norm_ -> optimizer step. Gradient accumulation is
        rejected in __init__, so every run_step IS an optimizer boundary. Hooks
        fire per the DeepSpeed-path contract: before/after_backward around the
        backward, before_step/after_step around the optimizer step; ``_iter``
        counts optimizer steps.
        """
        self.before_step()
        self.optimizers.zero_grad()

        meta = data_sample.get("metainfo", None)
        if meta is None:
            # Fail here (with context) instead of inside FreeDeviceBufferHook,
            # which would otherwise crash on None["metainfo"] after leaking
            # the device-buffer slot.
            raise RuntimeError(
                "data_sample['metainfo'] is None after preprocessing — hooks "
                "(FreeDeviceBufferHook, metric logging) require metainfo; the "
                "preprocessor must always emit it."
            )

        if self.timers is not None:
            self.timers["fwd_start"].record()
        loss_dict, outputs = self.model(data_sample)
        if self.timers is not None:
            self.timers["fwd_end"].record()

        del outputs  # Need to free outputs before bkwd to avoid peaking memory
        # No 1/N scaling: one batch per optimizer step. FSDP2's default gradient
        # divide still handles the cross-rank mean.
        loss = loss_dict["step_loss"]
        # keep DETACHED GPU scalars — values materialize at logging
        # boundaries (log_loss_dict / hooks), see EpochBasedTrainer.run_step
        loss_dict_log = {
            k: (v.detach() if torch.is_tensor(v) else v)
            for k, v in loss_dict.items()
        }

        self.before_backward(data_sample=data_sample, loss_dict=loss_dict_log, outputs=None)
        outputs = None
        data_sample = None
        loss_dict = None

        if self.timers is not None:
            self.timers["bwd_start"].record()
        loss.backward()
        if self.timers is not None:
            self.timers["bwd_end"].record()

        # after_backward: grads exist, optimizer has NOT stepped. With
        # accumulation off, FreeDeviceBufferHook is a no-op here (it frees in
        # after_step) — kept for the other hooks that use this callback.
        self.after_backward(
            data_sample={"metainfo": meta}, outputs=None, loss_dict=loss_dict_log
        )

        from torchtitan.distributed import utils as dist_utils  # lazy

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.max_norm,
            foreach=True,
        )
        # detached tensor: materialized at the logging boundary, no sync here
        self.event_recorder.put_scalar("grad_norm", grad_norm.detach())

        if self.timers is not None:
            self.timers["step_start"].record()
        self.optimizers.step()
        if self.timers is not None:
            self.timers["step_end"].record()
        # LR/WD hooks gate on this; every run_step IS an optimizer boundary here
        self._at_accumulation_boundary = True

        if self.metrics_processor is not None:
            metrics, _ = self.metrics_processor.process(
                data_sample={"metainfo": meta}, loss_dicts=[loss_dict_log]
            )
            for name, value in metrics.items():
                self.event_recorder.put_scalar(name, value)

        self.after_step(
            data_sample={"metainfo": meta},
            outputs=None,
            loss_dict=loss_dict_log,
        )
        self._iter += 1


class TestTrainer(BaseTrainer):
    """
    Test loop that runs ``model.predict`` on a held-out dataset and feeds the
    postprocessed predictions into a :class:`DatasetEvaluator`. Metrics are
    computed on what the model actually predicts (not on raw forward outputs /
    losses), so the evaluator sees tensors in the same space as the targets.

    Aligned with :class:`Inferencer`: plain ``torch`` (no DeepSpeed). Rationale:
      * ZeRO-1/2 shard optimizer state / gradients which do not exist at
        inference time -> zero memory benefit, just adds engine overhead and
        complicates checkpoint loading.
      * ZeRO-3 shards parameters, which can help fit large models on a single
        GPU, but the correct entrypoint for that is
        ``deepspeed.init_inference()`` (not ``deepspeed.initialize``). Deferred
        until a model actually requires sharded inference; tracked as a TODO.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.ray_context = get_context()
        self.event_recorder._iter = 0
        self.event_recorder._epoch = 0
        self._iter, self._epoch, self._val_iter = 0, 0, 0
        self.start_iter, self.start_epoch = 0, 0
        # TODO: refactor hooks to avoid flag
        self.with_grad_accumulation = False

        # initialize dataset and dataloader
        self.test_dataloader, _, _, self.host_buffer_actor, self.device_buffer, _ = get_dataloader(cfg)

        # initialize model
        model = REGISTRY.build("model", cfg.models.model, cfg)

        # NOT calling model._init_model_weights here: every meta-arch already
        # runs it in its own __init__ (jepa/maskedautoencoder/plainDETR/...),
        # and the checkpoint load below overwrites the weights anyway -- the
        # second init only wasted startup time on large models.
        # (EpochBasedTrainer likewise relies on the in-model init.)

        self.preprocessor = REGISTRY.build("preprocessor", cfg.datasets.preprocessor.name, cfg.datasets.preprocessor)

        # initialize checkpoint manager and load weights into the unwrapped
        # model (no DeepSpeed engine wrapper at inference time, so we load
        # directly into the nn.Module).
        self.checkpoint_manager = instantiate(
            cfg.checkpoint.checkpoint_manager,
            model=model,
        )
        self.checkpoint_meta = self.checkpoint_manager.load_for_eval(job_type="test")

        # torch backend flags (cudnn benchmark, tf32, etc.) and optional
        # torch.compile. Activation checkpointing is intentionally skipped:
        # it only saves memory during backward, which we never run here.
        if cfg.optimizations is not None:
            enable_optimizations(cfg=cfg)
        opt_cfg = get_model_optimizations_node(cfg)
        if opt_cfg.torch_compile.enable:
            logger.info("[TestTrainer] Applying torch.compile...")
            model = apply_compile(opt_cfg, model)

        # checkpoint_manager.load uses map_location=cpu; align device with the
        # dataloader (CUDA tensors). Mirrors Inferencer.
        _test_dev = torch.device("cuda", local_rank())
        self.model = model.to(_test_dev)
        self.model.eval()

        # initialize evaluator
        self.evaluator = REGISTRY.build("evaluator", cfg.evaluation.evaluator.name, cfg.evaluation.evaluator)


    def test(self):
        """
        Run model testing: iterate the test dataloader, call
        ``model.evaluate_step`` per step, and aggregate metrics via the evaluator.
        """
        self.before_test()

        with inference_context(self.model):
            with torch.no_grad():
                end = time.perf_counter()
                for idx, data_sample in enumerate(self.test_dataloader):
                    data_time = time.perf_counter() - end
                    data_sample = self.preprocessor(data_sample=data_sample, data_time=data_time, idx=idx)
                    self.run_test_step(idx, data_sample)
                    end = time.perf_counter()

        metrics = self.evaluator.evaluate()
        self.event_recorder.put_scalars(
            prefix="test",
            category="loss",
            scope="epoch",
            **{k: (v.item() if torch.is_tensor(v) else v)
                for k, v in metrics.items()
            }
        )
        self.evaluator.reset()

        self.after_test()

    def run_test_step(self, idx: int, data_sample: Sequence[dict]) -> None:
        """
        Iterate one test step: run ``model.evaluate_step`` and feed its
        postprocessed outputs to the evaluator. ``loss_dict`` is ``None`` because
        ``evaluate_step`` does not compute losses; evaluators that require losses
        must guard on this.
        """
        self.before_test_step()

        preds = self.model.evaluate_step(data_sample)
        self.evaluator.process(data_sample, preds, loss_dict=None)

        # for short testing runs:
        # if idx > 25:
        #     raise RuntimeError(
        #         f"Test stopped at step {idx} for debugging."
        #     )

        # only forward metainfo to hooks to avoid keeping prediction tensors
        # alive across step boundaries
        self.after_test_step(
            data_sample={"metainfo": data_sample.get("metainfo")},
            outputs=None,
            loss_dict=None,
        )
        self._iter += 1


class Inferencer(BaseTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.ray_context = get_context()
        self.event_recorder._iter = 0
        self.event_recorder._epoch = 0
        self._iter, self.start_iter, self.start_epoch = 0, 0, 0
        # TODO: refactor hooks to avoid flag
        self.with_grad_accumulation = False

        # initialize dataset and dataloader
        self.test_dataloader, _, dataloader_config, self.host_buffer_actor, self.device_buffer, database_df = get_dataloader(cfg)

        self.steps_per_epoch, val_steps_per_epoch = get_steps_per_epoch(
            config=cfg
        )

        # initialize model
        ray.logger.info(f"Building model {cfg.models.model!r} via REGISTRY")
        model = REGISTRY.build("model", cfg.models.model, cfg)

        # NOT calling model._init_model_weights here: every meta-arch already
        # runs it in its own __init__ (jepa/maskedautoencoder/plainDETR/...),
        # and the checkpoint load below overwrites the weights anyway -- the
        # second init only wasted startup time on large models.
        # (EpochBasedTrainer likewise relies on the in-model init.)

        ray.logger.info("initializing preprocessor...")
        self.preprocessor = REGISTRY.build("preprocessor", cfg.datasets.preprocessor.name, cfg.datasets.preprocessor)

        # initialize checkpoint manager and
        # load model state from checkpoint
        ray.logger.info("initializing checkpoint manager...")
        self.checkpoint_manager = instantiate(
            cfg.checkpoint.checkpoint_manager,
            model=model
        )
        self.checkpoint_meta = self.checkpoint_manager.load_for_eval(job_type="predict")

        # TODO: add support for multi-GPU inference via deepspeed.init_inference()
        #       when model sharding / tensor parallelism is needed.

        # torch backend flags (cudnn benchmark, tf32, etc.)
        if cfg.optimizations is not None:
            enable_optimizations(cfg=cfg)
        opt_cfg = get_model_optimizations_node(cfg)
        if opt_cfg.torch_compile.enable:
            logger.info("[Inferencer] Applying torch.compile...")
            model = apply_compile(opt_cfg, model)

        # Checkpoint load uses map_location=cpu; dataloader tensors are CUDA — align model device.
        _infer_dev = torch.device("cuda", local_rank())
        model = model.to(_infer_dev)

        self.model = model
        self.model.eval()

        # Build Output Memory Pools
        ray.logger.info("initializing buffer manager...")
        self.buffer_manager = BufferManager(
            local_rank=local_rank(),
            global_rank=process_rank(),
            node_id=node_id(),
            numa_node=torch_gpu_to_numa(local_rank())["numa_node"],
            rank_memory_budget_gb=cfg.datasets.buffers.rank_memory_budget_gb,
            max_concurrent_calls=cfg.datasets.buffers.max_concurrent_calls,
            safety_margin=cfg.datasets.buffers.safety_margin,
        )

        model_output_metadata = self.model.get_output_metadata()
        with open_dict(cfg.inference.inferencer_worker.outputs_metadata):
            cfg.inference.inferencer_worker.outputs_metadata.merge_with(model_output_metadata)            
        # Tile-mode inference restores predictions to the full original tile, so the
        # dense save/viz buffers must be sized for that. init_output_memory_pools owns
        # the enlargement; here we only feed it the DB stats (None => full-tile model).
        _restore_stats = getattr(
            dataloader_config.get("sample_store_desc", None), "stats", None
        ) if isinstance(dataloader_config, dict) else None
        ray.logger.info(f"Inference outputs_metadata merged:\n{cfg.inference.inferencer_worker.outputs_metadata}")

        ray.logger.info("initializing output memory pools...")
        init_output_memory_pools(
            buffer_manager=self.buffer_manager,
            output_metadata=cfg.inference.inferencer_worker.outputs_metadata,
            batch_size=cfg.clusters.batch_size_per_gpu,
            save=cfg.inference.inferencer_worker.save_outputs,
            viz=cfg.inference.inferencer_worker.vizualize_outputs,
            save_buffer_capacity=cfg.datasets.buffer_capacity,
            viz_buffer_capacity=cfg.datasets.buffer_capacity,
            layout=cfg.dataset_layout_order.upper(),
            restore_stats=_restore_stats,
        )
        # Both workers read/write this rank's shared-memory pools — they MUST be
        # scheduled on the same node as this process (soft=False: fail loud
        # rather than land on a node where the SHM segments don't exist).
        _worker_strategy = NodeAffinitySchedulingStrategy(node_id=node_id(), soft=False)

        kill_stale_actor(f"save_worker_rank_{process_rank()}")
        kill_stale_actor(f"viz_worker_rank_{process_rank()}")

        if cfg.inference.inferencer_worker.save_outputs:
            ray.logger.info("initializing save worker...")
            self.save_worker = SaveWorker.options(
                name=f"save_worker_rank_{process_rank()}",
                scheduling_strategy=_worker_strategy,
                # Reserve CPUs matching the writer thread pool; the default
                # num_cpus=1 lets Ray co-schedule past the actual parallelism.
                num_cpus=max(1, int(cfg.inference.save_worker.max_workers)),
            ).remote(
                buffer_manager=self.buffer_manager,
                max_workers=cfg.inference.save_worker.max_workers,
                save_mode=cfg.inference.save_worker.save_mode,
                chunk_spatial_shape=cfg.inference.save_worker.chunk_spatial_shape,
                shard_spatial_shape=cfg.inference.save_worker.shard_spatial_shape,
            )
        else:
            self.save_worker = None
        if cfg.inference.inferencer_worker.vizualize_outputs:
            ray.logger.info("initializing viz worker...")
            self.viz_worker = VizWorker.options(
                name=f"viz_worker_rank_{process_rank()}",
                scheduling_strategy=_worker_strategy,
            ).remote(
                buffer_manager=self.buffer_manager,
                output_dir=cfg.inference.viz_worker.output_dir,
                handler_configs=cfg.inference.viz_worker.handler_configs,
                max_workers=cfg.inference.viz_worker.max_workers,
            )
        else:
            self.viz_worker = None

        ray.logger.info("initializing inferencer worker...")
        _tp_cfg = OmegaConf.select(cfg, "datasets.timepoint_list")
        _timepoint_idxs_for_save = (
            [int(x) for x in _tp_cfg] if _tp_cfg is not None else None
        )
        self.inferencer_worker: InferencerWorker = REGISTRY.build(
            "inferencer",
            cfg.inference.inferencer_worker.name,
            cfg.inference.inferencer_worker,
            model=self.model,
            buffer_manager=self.buffer_manager,
            save_worker=self.save_worker,
            viz_worker=self.viz_worker,
            model_name=self.checkpoint_meta["model_name_slug"],
            timepoint_idxs_for_save=_timepoint_idxs_for_save,
        )

    def predict(self):
        """
        Run Model prediction.
        """
        self.before_test()

        # Teardown must run even when the predict loop raises: the workers are
        # detached named actors and the SHM pools outlive the driver, so an
        # exception without this finally leaked actors + segments across runs.
        try:
            with inference_context(self.model):
                with torch.no_grad():
                    end = time.perf_counter()
                    for idx, data_sample in enumerate(self.test_dataloader):
                        data_time = time.perf_counter() - end
                        data_sample = self.preprocessor(data_sample=data_sample, data_time=data_time, idx=idx)
                        self.run_inference_step(idx, data_sample)
                        end = time.perf_counter()
            # Success path: drain outstanding saves (finalize raises on dropped/
            # failed saves -- "output is INCOMPLETE"), THEN dispatch after_test
            # while the detached actors are still alive: InferenceMetricsHook
            # ray.gets save/viz worker metrics and would hit RayActorError on
            # killed actors. A loop or finalize raise skips after_test (as
            # before) and still tears down via the finally.
            self.inferencer_worker.finalize()
            self.after_test()
        finally:
            self._teardown()

    def _teardown(self) -> None:
        """Actor + SHM teardown; runs ALWAYS (predict()'s finally).

        Tear down detached actors explicitly. They are created with
        lifetime="detached" + stable names, so they are not garbage
        collected when this driver exits and would leak across runs.
        """
        if self.save_worker is not None:
            try:
                ray.kill(self.save_worker)
            except Exception as e:
                ray.logger.warning(f"Failed to kill save_worker: {e}")
        if self.viz_worker is not None:
            try:
                ray.kill(self.viz_worker)
            except Exception as e:
                ray.logger.warning(f"Failed to kill viz_worker: {e}")
        try:
            close = getattr(self.inferencer_worker, "close", None)
            if close is not None:
                close()          # release pinned staging buffers while streams are alive
            self.buffer_manager.shutdown()
        except Exception as e:
            ray.logger.warning(f"Failed to shut down buffer manager: {e}")

    def run_inference_step(
        self,
        idx: int,
        data_sample: Sequence[dict],
    ) -> None:
        """
        Iterate one prediction step.
        """
        self.before_test_step()

        self.inferencer_worker.predict(data_sample=data_sample)

        self.after_test_step(data_sample=data_sample, outputs=None, loss_dict=None)
        self._iter += 1


