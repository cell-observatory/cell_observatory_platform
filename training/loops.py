"""
Adopted with Apache License 2.0 from
https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/train_loop.py
https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
"""

import os
import time
import logging
import weakref
from pathlib import Path
from typing import Any, List, Optional, Sequence, Dict, Sequence

from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import get_class, instantiate, get_method

if not OmegaConf.has_resolver("now"):
    OmegaConf.register_new_resolver("now", lambda fmt: time.strftime(fmt))

import torch
from deepspeed import initialize
from ray.train import get_context
import ray

from cell_observatory_platform.training.helpers import (
    enable_optimizations,
    get_masked_input_data,
    get_steps_per_epoch,
    resume_run,
    summarize_model,
    configure_torch_comm_env,
    apply_compile,
    apply_activation_checkpointing,
    load_model_from_ckpt,
    aggregate_microbatch_losses,
    get_model_optimizations_node
)
from cell_observatory_platform.training.hooks import HookBase
from cell_observatory_platform.data.data_types import TORCH_DTYPES
from cell_observatory_platform.data.dataloaders import get_dataloader
from cell_observatory_platform.parallelism.utils import get_cp_buffers
from cell_observatory_platform.training.loggers import EventRecorder, MetricsProcessor
from cell_observatory_platform.training.optimizers import get_optimizer, build_optimizers
from cell_observatory_platform.data.datasets.pretrain_dataset_ray import get_dataloader_ray
from cell_observatory_platform.utils.context import (
    inference_context,
    process_rank,
    get_world_size,
    local_rank,
    node_id,
    torch_gpu_to_numa,
)
from cell_observatory_platform.inference.inferencer import InferencerWorker
from cell_observatory_platform.training.schedulers import get_param_groups, get_schedulers, build_lr_schedulers, build_wd_schedulers
from cell_observatory_platform.data.datasets.buffers import BufferManager, init_output_memory_pools
from cell_observatory_platform.inference.saver import SaveWorker
from cell_observatory_platform.inference.visualizer import VizWorker

from torchtitan.tools import utils
from torchtitan.components.ft import FTManager
from torchtitan.distributed import ParallelDims 
from torchtitan.distributed import utils as dist_utils
from torchtitan.components.checkpoint import CheckpointManager

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
                         f"Expected 'train' or 'test', got '{config.job_type}'.")
    
    return {"best_metric": trainer_per_worker.best_metric 
            if hasattr(trainer_per_worker, 'best_metric') else None}


class BaseTrainer:
    """
    Base class for iterative trainer with hooks.
    """

    def __init__(self, config: DictConfig) -> None:
        # initialize event recorder
        self.event_recorder: EventRecorder = instantiate(_ensure_full_path(config.loggers.event_recorder))

        # initialize event_writers
        event_writers = self._build_event_writers(w_cfgs=config.loggers.event_writers, recorder=self.event_recorder)
        self.event_writers_list = instantiate(
            _ensure_full_path(config.loggers.event_writers_list), writers=event_writers
        )

        # intialize hooks
        hooks = self._build_hooks(config.hooks.hooks_list, self.event_writers_list)
        self._hooks: List[HookBase] = []
        self.register_hooks(hooks)

    @staticmethod
    def _build_event_writers(w_cfgs, recorder):
        writers = []
        for writer_cfg in w_cfgs:
            writer = instantiate(_ensure_full_path(writer_cfg), event_recorder=recorder)
            writers.append(writer)
        return writers

    @staticmethod
    def _build_hooks(h_cfgs, event_writers):
        hooks = []
        for hc in h_cfgs:
            if hc._target_ and not hc._target_.startswith("cell_observatory_platform."):
                hc._target_ = f"cell_observatory_platform.{hc._target_}"

            # inject writers into PeriodicWriter hook
            if hc._target_.endswith(".PeriodicWriter"):
                hook = instantiate(hc, writers=event_writers)
            else:
                hook = instantiate(hc)
            hooks.append(hook)
        return hooks

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        """
        Register hooks to the trainer. The hooks are executed
        in the order they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        allowed_subclasses = [h.__name__ for h in HookBase.__subclasses__()]
        hooks = [h for h in hooks if h is not None]

        for h in hooks:
            assert type(h).__name__ in allowed_subclasses
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


class EpochBasedTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        print(f"Run Config:\n{OmegaConf.to_yaml(cfg)}")

        self.ray_context = get_context()
        self.val_begin, self.val_interval = cfg.evaluation.val_begin, cfg.evaluation.val_interval
        self.stop_training, self._max_epochs = False, cfg.schedulers.epochs

        dp_degree = get_world_size()
        # Verify batch sizes
        assert (
            cfg.clusters.batch_size % (cfg.clusters.batch_size_per_gpu * dp_degree) == 0
        ), (
            f"global batch size must be multiple of local batch size times "
            f"data-parallel degree ({cfg.clusters.batch_size} "
            f"% ({cfg.clusters.batch_size_per_gpu} * {dp_degree}) != 0)"
        )
        # Calculate gradient accumulation steps
        self.gradient_accumulation_steps = cfg.clusters.batch_size // (
            cfg.clusters.batch_size_per_gpu * dp_degree
        )
        assert (
            cfg.deepspeed.gradient_accumulation_steps == self.gradient_accumulation_steps
        ), "Deepspeed gradient_accumulation_steps does not match the calculated value."
        self.with_grad_accumulation = self.gradient_accumulation_steps > 1
        assert self.gradient_accumulation_steps > 0, "Calculated gradient accumulation steps must be > 0."
        # NOTE: turn off gradient accumulation for now while debugging
        assert self.gradient_accumulation_steps == 1, "Gradient accumulation currently not supported."

        # initialize dataset and dataloader
        _, _, self.dataloader_config, self.host_buffer_actor, self.device_buffer, _ = get_dataloader(cfg)

        self.steps_per_epoch, self.val_steps_per_epoch = get_steps_per_epoch(
            config=cfg,
            gradient_accumulation_steps=self.gradient_accumulation_steps
        )

        # initialize model
        BUILD = get_method(cfg.models.BUILD)
        model = BUILD(cfg)

        with torch.no_grad():
            model.init_model_weights(buffer_device="cuda")

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

        self.preprocessor = instantiate(cfg.datasets.preprocessor)

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
            self.checkpoint_manager.load()

        if cfg.optimizations.with_model_summary:
            rank = process_rank()
            if rank == 0:
                input_shape = (cfg.clusters.batch_size, *cfg.datasets.input_shape)
                input_data = get_masked_input_data(model, input_shape)

                summarize_model(
                    model=model,
                    inputs=input_shape,
                    input_data=input_data,
                    batch_size=cfg.clusters.batch_size,
                    logdir=cfg.paths.outdir,
                )

        # initialize optimizer and learning rate scheduler
        param_groups = get_param_groups(cfg, model)
        self.opt, _ = get_optimizer(
            params=param_groups,
            config=cfg,
            optimizer=cfg.optimizers.opt,
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

        # initialize deepspeed
        self.model, self.optimizers, _, _ = initialize(
            model=model,
            optimizer=self.opt,
            config=OmegaConf.to_container(cfg.deepspeed, resolve=True)
        )
        self.checkpoint_manager.model = self.model

        if cfg.checkpoint.checkpoint_manager.resume_checkpointdir:
            self.checkpoint_manager.load()

        # if resume job, gather the state from the checkpoint
        # else intialize outdir, logdir, and checkpointdir
        # these directories should be empty if not resuming a job
        # to avoid overwriting existing checkpoints
        best_metric, step, epoch = resume_run(self, cfg)
        self.start_epoch, self.start_iter, self.best_metric = epoch, step, best_metric
        self._epoch, self._iter, self._val_iter, self._curr_val_metric = self.start_epoch, self.start_iter, 0, float('inf')

        if self.start_iter > 0:
            logger.info("[Trainer] Resuming training without loading previous optimizer state.")
            logger.info(f"[Trainer] Fast forwarding lr and wd schedulers to iter {self.start_iter} and epoch {self.start_epoch}.")
            # fast forward lr and wd schedulers to the correct step
            # TODO: consider making more flexible
            if self.wd_schedulers is not None:
                for _ in range(self.start_iter):
                    self.wd_schedulers.step()
            
            if self.schedulers.update_type == "epoch":
                for epoch in range(self.start_epoch):
                    self.schedulers.step(epoch)
            elif self.schedulers.update_type == "step":
                for iter in range(self.start_iter):
                    self.schedulers.step(iter)
            else:
                raise NotImplementedError(f'{self.schedulers.update_type=} is not supported')

        # initialize evaluator
        self.evaluator = instantiate(cfg.evaluation.evaluator)

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

        end = time.perf_counter()
        for idx, data_sample in enumerate(train_dataloader):
            data_time = time.perf_counter() - end
            data_sample = self.preprocessor(data_sample=data_sample, data_time=data_time)
            # run one step with the fetched data sample
            self.run_step(idx, data_sample)
            end = time.perf_counter()

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
        loss_dict, outputs = self.model(data_sample)

        del outputs  # Need to free outputs before bkwd to avoid peaking memory
        loss = loss_dict["step_loss"]
        loss_dict_log = {
            k: (v.detach().float().cpu().item() if torch.is_tensor(v) else v)
            for k, v in loss_dict.items()
        }

        outputs = None
        data_sample = None
        loss_dict = None

        self.model.backward(loss)
        self.model.step()

        # for short testing runs:
        # if idx > 25:
        #     raise RuntimeError(
        #         f"Training stopped at step {idx} for testing."
        #     )
        # logger.info(f"step_loss: {loss_dict['step_loss']}, lr: {self.opt.param_groups[0]['lr']}")

        self.after_step(
            data_sample={"metainfo": meta} if meta is not None else None,
            outputs=None,
            loss_dict=loss_dict_log,
        )
        self._iter += 1

    def run_validation(self, val_dataloader) -> None:
        """
        Run validation.
        """
        self.before_validation()
        # technically, contexts could be a hook
        # but kept here for clarity
        with inference_context(self.model):
            with torch.no_grad():
                end = time.perf_counter()
                for idx, data_sample in enumerate(val_dataloader):
                    data_time = time.perf_counter() - end
                    data_sample = self.preprocessor(data_sample=data_sample, data_time=data_time)
                    # run one step with the fetched data sample
                    self.run_validation_step(idx, data_sample)
                    end = time.perf_counter()

        metrics = self.evaluator.evaluate()
        self.event_recorder.put_scalars(
            scope="epoch", prefix="val_", **{k: (v.item() if torch.is_tensor(v) else v) for k, v in metrics.items()}
        )
        self.evaluator.reset()

        self.after_validation()

    def run_validation_step(self, idx: int, data_sample: Sequence[dict]) -> None:
        """
        Iterate one validation step.
        """
        self.before_val_step()

        loss_dict, outputs = self.model(data_sample)
        loss_dict_log = {
            k: (v.detach().float().cpu().item() if torch.is_tensor(v) else v)
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

    # def run(self):
    #     """
    #     Launch training.
    #     """
    #     self.before_train()

    #     while self._epoch < self._max_epochs and not self.stop_training:
    #         self.run_epoch()

    #     self.after_train()

    # def run_epoch(self) -> None:
    #     """
    #     Iterate one epoch.
    #     """
    #     self.before_epoch()
        
    #     self.train_dataloader_iter = iter(self.train_dataloader)

    #     for _ in range(self.steps_per_epoch):
    #         self.run_step()

    #     if self.val_dataloader and \
    #        (self._epoch >= self.val_begin and
    #         (self._epoch - self.val_begin) % self.val_interval == 0):
    #         # run validation
    #         self.run_validation()

    #     self.after_epoch()
    #     self._epoch += 1

    # def run_step(self) -> None:
    #     """
    #     Iterate one step.
    #     """
    #     self.before_step()
        
    #     microbatch_loss_dicts = []
    #     for _microbatch in range(self.gradient_accumulation_steps):
    #         t_start = time.perf_counter()
    #         data_sample = next(self.train_dataloader_iter)
    #         data_time = time.perf_counter() - t_start
    #         data_sample = self.preprocessor(data_sample=data_sample, data_time=data_time)

    #         # we enforce that all models compute
    #         # their losses in the forward pass
    #         # and return a loss_dict with at least
    #         # a "step_loss" key together with
    #         # the outputs of the model
    #         loss_dict = self.forward_backward_step(self._iter, data_sample)
    #         microbatch_loss_dicts.append({
    #             k: (v.detach() if torch.is_tensor(v) else v)
    #             for k, v in loss_dict.items()
    #         })

    #         self.model.step()

    #     # for short testing runs:
    #     # if idx > 25:
    #     #     raise RuntimeError(
    #     #         f"Training stopped at step {idx} for testing."
    #     #     )
    #     # logger.info(f"step_loss: {loss_dict['step_loss']}, lr: {self.optimizers.param_groups[0]['lr']}")

    #     aggregated_loss = aggregate_microbatch_losses(microbatch_loss_dicts, self.gradient_accumulation_steps)
    #     self.after_step(data_sample=data_sample, outputs=None, loss_dict=aggregated_loss)
    #     self._iter += 1

    # def forward_backward_step(self, idx, data_sample: Sequence[dict]):
    #     """
    #     Iterate one mini-batch fwd+bkwd step.
    #     """
    #     loss_dict, outputs = self.model(data_sample)
    #     # Need to free outputs before bwd to avoid peaking memory
    #     del outputs
    #     self.model.backward(loss_dict["step_loss"])
        
    #     self.after_backward(data_sample=data_sample, loss_dict=loss_dict, outputs=None)
        
    #     return loss_dict

    # def run_validation(self) -> None:
    #     """
    #     Run validation.
    #     """
    #     self.before_validation()

    #     self.val_dataloader_iter = iter(self.val_dataloader)

    #     # technically, contexts could be a hook
    #     # but kept here for clarity
    #     with inference_context(self.model):
    #         with torch.no_grad():
    #             for step_idx in range(self.val_steps_per_epoch):
    #                 self.run_validation_step(step_idx)

    #     metrics = self.evaluator.evaluate()
    #     self.event_recorder.put_scalars(
    #         scope="epoch",
    #         prefix="val_",
    #         **{k: (v.item() if torch.is_tensor(v) else v)
    #             for k, v in metrics.items()
    #         }
    #     )
    #     self.evaluator.reset()

    #     self.after_validation()
    
    # def run_validation_step(self, idx: int) -> None:
    #     """
    #     Iterate one validation step.
    #     """
    #     self.before_val_step()

    #     microbatch_loss_dicts = []
    #     for _microbatch in range(self.gradient_accumulation_steps):
    #         t_start = time.perf_counter()
    #         data_sample = next(self.val_dataloader_iter)
    #         data_time = time.perf_counter() - t_start
    #         data_sample = self.preprocessor(
    #             data_sample=data_sample,
    #             data_time=data_time,
    #         )

    #         loss_dict, outputs = self.validation_forward_step(data_sample)
    #         microbatch_loss_dicts.append({
    #             k: (v.detach() if torch.is_tensor(v) else v)
    #             for k, v in loss_dict.items()
    #         })

    #         self.evaluator.process(data_sample, outputs, loss_dict)

    #     aggregated_loss = aggregate_microbatch_losses(microbatch_loss_dicts, self.gradient_accumulation_steps)
    #     self.after_val_step(data_sample=data_sample, outputs=None, loss_dict=aggregated_loss)
    #     self._val_iter += 1

    # def validation_forward_step(self, data_sample: Sequence[dict]):
    #     """
    #     Run microbatch for validation.
    #     """
    #     loss_dict, outputs = self.model(data_sample)
    #     return loss_dict, outputs


class TestTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        
        self.ray_context = get_context()
        self.event_recorder._iter = 0
        self.event_recorder._epoch = 0
        self._iter, self.start_iter, self.start_epoch = 0, 0, 0
        # TODO: refactor hooks to avoid flag
        self.with_grad_accumulation = False

        # initialize dataset and dataloader
        self.test_dataloader, _, _, self.host_buffer_actor, self.device_buffer, database_df = get_dataloader(cfg)

        self.steps_per_epoch, val_steps_per_epoch = get_steps_per_epoch(
            config=cfg
        )

        # initialize model
        BUILD = get_method(cfg.models.BUILD)
        model = BUILD(cfg)

        with torch.no_grad():
            model.init_model_weights(buffer_device="cuda")

        self.preprocessor = instantiate(cfg.datasets.preprocessor)

        # initialize checkpoint manager and
        # load model state from checkpoint
        self.checkpoint_manager = instantiate(
            cfg.checkpoint.checkpoint_manager,
            model=model
        )
        self.checkpoint_manager.load()

        # initialize optimizer (needed for deepspeed init)
        self.opt, _ = get_optimizer(
            params=model.parameters(),
            config=cfg,
            optimizer=cfg.optimizers.opt,
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

        # initialize deepspeed
        self.model, self.optimizers, _, _ = initialize(
            model=model,
            optimizer=self.opt,
            config=OmegaConf.to_container(cfg.deepspeed, resolve=True)
        )

        # initialize evaluator
        self.evaluator = instantiate(cfg.evaluation.evaluator)

    def test(self):
        """
        Run Model testing.
        """
        self.before_test()

        with inference_context(self.model):
            with torch.no_grad():
                end = time.perf_counter()
                for idx, data_sample in enumerate(self.test_dataloader):
                    data_time = time.perf_counter() - end
                    data_sample = self.preprocessor(data_sample=data_sample, data_time=data_time)
                    self.run_test_step(idx, data_sample)
                    end = time.perf_counter()
        
        metrics = self.evaluator.evaluate()
        self.event_recorder.put_scalars(
            prefix="evaluator_",
            scope="epoch",
            **{k: (v.item() if torch.is_tensor(v) else v)
                for k, v in metrics.items()
            }
        )
        self.evaluator.reset()

        self.after_test()
    
    def run_test_step(self, idx: int, data_sample: Sequence[dict]) -> None:
        """
        Iterate one test step.
        """
        self.before_test_step()

        loss_dict, outputs = self.model(data_sample)
        self.evaluator.process(data_sample, outputs, loss_dict)

        # for short testing runs:
        # if idx > 25:
        #     raise RuntimeError(
        #         f"Test stopped at step {idx} for debugging."
        #     )
        # logger.info(f"step_loss: {loss_dict['step_loss']}")

        self.after_test_step(data_sample=data_sample, 
                             outputs=outputs, loss_dict=loss_dict)
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
        self.test_dataloader, _, _, self.host_buffer_actor, self.device_buffer, database_df = get_dataloader(cfg)

        self.steps_per_epoch, val_steps_per_epoch = get_steps_per_epoch(
            config=cfg
        )

        # initialize model
        BUILD = get_method(cfg.models.BUILD)
        model = BUILD(cfg)

        with torch.no_grad():
            model.init_model_weights(buffer_device="cuda")

        self.preprocessor = instantiate(cfg.datasets.preprocessor)

        # initialize checkpoint manager and
        # load model state from checkpoint
        self.checkpoint_manager = instantiate(
            cfg.checkpoint.checkpoint_manager,
            model=model
        )
        self.checkpoint_manager.load()

        # initialize optimizer
        self.opt, _ = get_optimizer(
            params=model.parameters(),
            config=cfg,
            optimizer=cfg.optimizers.opt,
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

        # initialize deepspeed
        self.model, self.optimizers, _, _ = initialize(
            model=model,
            optimizer=self.opt,
            config=OmegaConf.to_container(cfg.deepspeed, resolve=True)
        )

        # Build Output Memory Pools
        self.buffer_manager = BufferManager(
            local_rank=local_rank(),
            global_rank=process_rank(),
            node_id=node_id(),
            numa_node=torch_gpu_to_numa(local_rank())["numa_node"],
            rank_memory_budget_gb=cfg.inference.memory.rank_memory_budget_gb,
            max_concurrent_calls=cfg.inference.buffer_manager.max_concurrent_calls,
            safety_margin=cfg.inference.buffer_manager.safety_margin,
        )
        
        init_output_memory_pools(
            buffer_manager=self.buffer_manager,
            output_metadata=cfg.inference.outputs_metadata, #TODO: make sure this includes aux outputs, targets, and inputs if we want to vizualize them
            batch_size=cfg.clusters.batch_size_per_gpu,
            save=cfg.inference.save,
            viz=cfg.inference.viz,
            save_buffer_capacity=cfg.inference.buffer_manager.save_buffer_capacity,
            viz_buffer_capacity=cfg.inference.buffer_manager.viz_buffer_capacity,
        )
        if cfg.inference.save:
            self.save_worker = SaveWorker.options(name=f"save_worker_rank_{process_rank()}").remote(
                buffer_manager=self.buffer_manager,
                max_retries=cfg.inference.max_retries,
                retry_backoff_s=cfg.inference.retry_backoff_s,
            )
        else:
            self.save_worker = None
        if cfg.inference.viz:
            self.viz_worker = VizWorker.options(name=f"viz_worker_rank_{process_rank()}").remote(
                buffer_manager=self.buffer_manager
            )
        else:
            self.viz_worker = None
        # initialize inferencer_worker
        self.inferencer_worker = instantiate(
            cfg.inference, 
            model=self.model, 
            database=database_df,
            buffer_manager=self.buffer_manager,
            save_worker=self.save_worker,
            viz_worker=self.viz_worker,
        )

    def predict(self):
        """
        Run Model prediction.
        """
        self.before_test()

        with inference_context(self.model):
            with torch.no_grad():
                end = time.perf_counter()
                for idx, data_sample in enumerate(self.test_dataloader):
                    data_time = time.perf_counter() - end
                    data_sample = self.preprocessor(data_sample=data_sample, data_time=data_time)
                    self.run_inference_step(idx, data_sample)
                    end = time.perf_counter()
        
        self.inferencer_worker.finalize()

        self.after_test()

    def run_inference_step(
        self,
        idx: int,
        data_sample: Sequence[dict],
        save: bool = False,
        viz: bool = False
    ) -> None:
        """
        Iterate one prediction step.
        """
        self.before_test_step()

        self.inferencer_worker.predict(data_sample=data_sample)

        self.after_test_step(data_sample=data_sample, outputs=None, loss_dict=None)
        self._iter += 1

class ParallelEpochBasedTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.cfg = cfg

        print(f"Run Config:\n{OmegaConf.to_yaml(cfg)}")
        
        if cfg.optimizations is not None:
            enable_optimizations(cfg=cfg)

        self.dtype = TORCH_DTYPES[cfg.quantization].value \
            if isinstance(cfg.quantization, str) else cfg.quantization
        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        # Device has to be set before creating TorchFT manager
        device_module.set_device(self.device)

        self.ray_context = get_context()
        self.val_begin, self.val_interval = cfg.evaluation.val_begin, cfg.evaluation.val_interval
        self.stop_training, self._max_epochs, self.num_tokens_seen = False, cfg.schedulers.epochs, 0

        # Init distributed and build meshes
        self.parallel_dims = parallel_dims = self.init_distributed(cfg)

        world_mesh = parallel_dims.world_mesh
        if parallel_dims.dp_enabled:
            dp_mesh = world_mesh["dp"]
            dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0

        # NOTE: we don't support TorchFT properly yet
        self.ft_manager = FTManager(ft_config=cfg.parallelism.fault_tolerance)
        dp_degree, dp_rank = self.ft_manager.get_dp_info(dp_degree, dp_rank)

        # Verify batch sizes
        assert (
            cfg.clusters.batch_size % (cfg.clusters.batch_size_per_gpu * dp_degree) == 0
        ), (
            f"global batch size must be multiple of local batch size times "
            f"data-parallel degree ({cfg.clusters.batch_size} "
            f"% ({cfg.clusters.batch_size_per_gpu} * {dp_degree}) != 0)"
        )

        # Calculate gradient accumulation steps
        self.gradient_accumulation_steps = cfg.clusters.batch_size // (
            cfg.clusters.batch_size_per_gpu * dp_degree
        )
        self.with_grad_accumulation = self.gradient_accumulation_steps > 1
        assert self.gradient_accumulation_steps > 0, "Calculated gradient accumulation steps must be > 0."

        # Initialize dataset and dataloader        
        self.train_dataloader_iter, self.val_dataloader_iter = None, None
        self.train_dataloader, self.val_dataloader, _, \
            self.host_buffer_actor, self.device_buffer, _ = get_dataloader(cfg, dp_degree, dp_rank)

        self.steps_per_epoch, self.val_steps_per_epoch = get_steps_per_epoch(
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            config=cfg,
            gradient_accumulation_steps=self.gradient_accumulation_steps
        )

        self._iter, self._epoch = 0, 0
        self._val_iter, self._curr_val_metric = 0, float("inf")

        if os.environ.get("RESTART", "FALSE").upper() == "TRUE":
            logger.info("RESTART flag detected. Resuming from latest checkpoint.")
            with open_dict(cfg):
                cfg.paths.resume_checkpointdir = Path(cfg.paths.outdir) / "checkpoints"
                cfg.checkpoint.checkpoint_manager.resume_checkpointdir = cfg.paths.resume_checkpointdir

        # Control garbage collection to avoid straggler effects
        self.gc_handler = utils.GarbageCollection(
            gc_freq=cfg.parallelism.gc.gc_freq, 
            debug=cfg.parallelism.gc.gc_debug
        )

        # Set random seed, and maybe enable deterministic mode
        dist_utils.set_determinism(
            world_mesh,
            self.device,
            cfg.parallelism.debug.seed,
            cfg.parallelism.debug.deterministic,
        )

        with (
            torch.device("meta"),
            utils.set_default_dtype(self.dtype),
        ):
            # Initialize model
            BUILD = get_method(cfg.models.BUILD)
            model = BUILD(cfg)

        if hasattr(model, "_get_nparams_and_flops"):
            self.model_param_count, self.num_flops_per_token = \
                model._get_nparams_and_flops(batch_size=cfg.clusters.batch_size_per_gpu, device="meta")
        else:
            logger.warning(
                "Model does not implement `_get_nparams_and_flops` method. "
                "Setting model_param_count and num_flops_per_token to -1."
                "Flops and parameter counts will be unavailable in reported metrics."
            )
            self.model_param_count, self.num_flops_per_token = -1, -1

        self.preprocessor = instantiate(cfg.datasets.preprocessor)

        self.seq_len = int(self.preprocessor.seq_len)
        assert (
            self.seq_len % parallel_dims.seq_len_divisor == 0
        ), f"""
            Sequence length {self.seq_len} currently must be divisible by the product 
            of TP degree ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
            See: https://github.com/pytorch/torchtitan/issues/1306
            """

        # move sharded model to CPU/GPU and initialize weights via DTensor
        if cfg.checkpoint.checkpoint_manager.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif cfg.parallelism.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = device_type
        else:
            init_device = device_type
            buffer_device = None

        if parallel_dims.pp_enabled:
            raise NotImplementedError("Pipeline parallelism is not yet supported.")
        else:
            # apply PT-D Tensor Parallel, activation checkpointing, 
            # torch.compile, Data Parallel
            PARALLEL = get_method(cfg.models.PARALLELISM)
            model = PARALLEL(model, parallel_dims, cfg)

            model.to_empty(device=init_device)
            with torch.no_grad():
                model.init_model_weights(buffer_device=buffer_device)
            model.train()

            self.model_parts = [model]

        self.ft_manager.maybe_set_all_reduce_hook(self.model_parts)

        # build optimizer after applying parallelisms to the model
        self.optimizers = build_optimizers(
            model_parts=self.model_parts,
            optimizer_config=cfg.optimizers,
            parallel_dims=parallel_dims,
            ft_manager=self.ft_manager,
        )
        self.schedulers = build_lr_schedulers(
            optimizers=self.optimizers,
            lr_scheduler_config=cfg.schedulers,
            training_steps=self.steps_per_epoch
        )
        self.wd_schedulers = build_wd_schedulers(
            optimizers=self.optimizers,
            wd_scheduler_config=cfg.schedulers.wd_scheduler,
            training_steps=self.steps_per_epoch,
        )

        # TODO: enable model converters hooks
        # Post optimizer step model converters hook.
        # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
        # where it issues a single all-reduce for all parameters at once for better performance
        # self.optimizers.register_step_post_hook(
        #     lambda *args, **kwargs: model_converters.post_optimizer_hook(
        #         self.model_parts
        #     )
        # )

        loss_parallel_enabled = (
            parallel_dims.tp_enabled
            and not cfg.parallelism.training.disable_loss_parallel
        )
        enable_compiled_autograd = cfg.optimizations.models.torch_compile.get(
            "enable_compiled_autograd"
        )
        self.train_context = dist_utils.get_train_context(loss_parallel_enabled, 
                                                          enable_compiled_autograd)
        self.maybe_enable_amp = dist_utils.maybe_enable_amp(
            parallel_dims,
            cfg.parallelism.training.mixed_precision_param,
            device_type,
        )

        # Initialize metric postprocessor and evaluator
        self.timers = self.init_timers()
        
        self.metrics_processor = MetricsProcessor(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            optimizers=self.optimizers,
            lr_schedulers=self.schedulers,
            model_parts=self.model_parts,
            parallel_dims=self.parallel_dims,
            timers=self.timers,
            num_flops_per_token=self.num_flops_per_token,
            model_param_count=self.model_param_count
        )
        self.evaluator = instantiate(cfg.evaluation.evaluator)

        # if resume job, gather the state from the checkpoint
        # else intialize outdir, logdir, and checkpointdir
        # these directories should be empty if not resuming a job
        # to avoid overwriting existing checkpoints
        best_metric, step, epoch = resume_run(self, cfg)
        self.start_epoch, self.start_iter, self.best_metric = epoch, step, best_metric
        self._epoch, self._iter, self._val_iter, self._curr_val_metric = self.start_epoch, self.start_iter, 0, float('inf')

        # initialize checkpoint manager
        # NOTE: any attributes must be initialized before checkpoint loading
        self.checkpoiunt_save_period = cfg.checkpoint.checkpoint_manager.save_period
        self.checkpoint_manager = CheckpointManager(
            dataloader=None,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.schedulers,
            states={"train_state": self, "wd_schedulers": self.wd_schedulers},
            checkpoint_config=cfg.checkpoint.checkpoint_manager,
            # TODO: support sd_adapter
            sd_adapter=None,
            base_folder=cfg.checkpoint.checkpoint_manager.save_checkpointdir,
            ft_manager=self.ft_manager,
        )

        if cfg.checkpoint.checkpoint_manager.resume_checkpointdir is not None or \
            cfg.checkpoint.checkpoint_manager.pretrained_checkpointdir is not None:
            if self.start_iter > 0 and Path(cfg.checkpoint.checkpoint_manager.resume_checkpointdir).exists():
                logger.info("[Trainer] Resuming training from step {} and epoch {}.".format(self.start_iter, self.start_epoch))
                self.checkpoint_manager.load(step=self.start_iter)
            elif Path(cfg.checkpoint.checkpoint_manager.pretrained_checkpointdir).exists():
                logger.info(
                    "[Trainer] Initializing model weights from pretrained checkpoint at %s. "
                    "Starting training from step 0 and epoch 0.",
                    cfg.checkpoint.checkpoint_manager.pretrained_checkpointdir,
                )
                load_model_from_ckpt(cfg, self.checkpoint_manager)
                # We force fresh run statistics after loading pretrained weights
                self.start_epoch, self.start_iter = 0, 0
                self._epoch, self._iter, self._val_iter = 0, 0, 0
                self.best_metric = float("inf")

    def init_distributed(self, cfg) -> ParallelDims:
        configure_torch_comm_env(cfg.parallelism.comm)
        return ParallelDims(
            dp_shard=cfg.parallelism.training.data_parallel_shard_degree,
            dp_replicate=cfg.parallelism.training.data_parallel_replicate_degree,
            cp=cfg.parallelism.training.context_parallel_degree,
            tp=cfg.parallelism.training.tensor_parallel_degree,
            pp=cfg.parallelism.training.pipeline_parallel_degree,
            ep=cfg.parallelism.training.expert_parallel_degree,
            etp=cfg.parallelism.training.expert_tensor_parallel_degree,
            world_size=get_world_size(),
        )
    
    def init_timers(self):
        self.fwd_start = torch.cuda.Event(enable_timing=True)
        self.fwd_end = torch.cuda.Event(enable_timing=True)
        self.bwd_start = torch.cuda.Event(enable_timing=True)
        self.bwd_end = torch.cuda.Event(enable_timing=True)
        self.step_start = torch.cuda.Event(enable_timing=True)
        self.step_end = torch.cuda.Event(enable_timing=True)
        timers = {             
            "fwd_start": self.fwd_start,
            "fwd_end": self.fwd_end,
            "bwd_start": self.bwd_start,
            "bwd_end": self.bwd_end,
            "step_start": self.step_start,
            "step_end": self.step_end,
        }
        return timers

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

        self.train_dataloader_iter = iter(self.train_dataloader)
        
        for _ in range(self.steps_per_epoch):
            self.run_step()

        if self.val_dataloader and \
           (self._epoch >= self.val_begin and
            (self._epoch - self.val_begin) % self.val_interval == 0):
            self.run_validation()

        self.after_epoch()
        self._epoch += 1

    def run_step(self) -> None:
        """
        Iterate one step.
        """
        self.before_step()

        self.optimizers.zero_grad()
        
        microbatch_loss_dicts = []
        for _microbatch in range(self.gradient_accumulation_steps):
            t_start = time.perf_counter()
            data_sample = next(self.train_dataloader_iter)
            data_time = time.perf_counter() - t_start
            data_sample = self.preprocessor(data_sample=data_sample, data_time=data_time)
            
            loss_dict = self.forward_backward_step(self._iter, data_sample)
            microbatch_loss_dicts.append({
                k: (v.detach() if torch.is_tensor(v) else v)
                for k, v in loss_dict.items()
            })

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.cfg.parallelism.training.max_norm,
            foreach=True,
            pp_mesh=(
                self.parallel_dims.world_mesh["pp"] if self.parallel_dims.pp_enabled else None
            ),
            ep_enabled=self.parallel_dims.ep_enabled,
        )
        
        self.checkpoint_manager.maybe_wait_for_staging()
        self.step_start.record()
        self.optimizers.step()
        self.step_end.record()

        # for short testing runs:
        # if self._iter > 25:
        #     raise RuntimeError(
        #         f"Training stopped at step {self._iter} for testing."
        #     )
        # logger.info(f"step_loss: {loss}")

        metrics, aggregated_loss = self.metrics_processor.process(
            data_sample=data_sample,
            loss_dicts=microbatch_loss_dicts,
            extra_metrics={
                "perf/grad_norm": grad_norm.item(),
            },
        )
        data_sample["advanced_metrics"] = metrics

        self.after_step(data_sample=data_sample, 
                        outputs=None, 
                        loss_dict=aggregated_loss)
        self._iter += 1

    def forward_backward_step(self, idx, data_sample: Sequence[dict]):
        """
        Iterate one mini-batch fwd+bkwd step.
        """
        # TODO: consider adding as step Hook in Training/Hooks.py
        # applies context parallelism if cp is enabled
        if self.parallel_dims.cp_enabled:
            cp_buffers, cp_seq_dims = get_cp_buffers(data_sample, 
                                                     self.model_parts, 
                                                     disable_load_balance=True)
            optional_context_parallel_ctx = (
                dist_utils.create_context_parallel_ctx(
                    cp_mesh=self.parallel_dims.world_mesh["cp"],
                    cp_buffers=cp_buffers,
                    cp_seq_dims=cp_seq_dims,
                    cp_no_restore_buffers={cp_buffers[0], cp_buffers[1]}, # inputs, targets
                    cp_rotate_method=self.cfg.parallelism.training.context_parallel_rotate_method,
                )
            )
        else:
            optional_context_parallel_ctx = None

        if self.parallel_dims.pp_enabled:
            raise NotImplementedError("Pipeline parallelism is not yet supported.")
        else:
            # Non-PP forward / backward
            with self.train_context(optional_context_parallel_ctx):
                assert len(self.model_parts) == 1
                with self.maybe_enable_amp:
                    self.fwd_start.record()
                    loss_dict, outputs = self.model_parts[0](data_sample)
                    self.fwd_end.record()
                    loss = loss_dict["step_loss"]
                # Need to free outputs before bwd to avoid peaking memory
                del outputs
                self.bwd_start.record()
                loss.backward()
                self.bwd_end.record()

        self.after_backward(data_sample=data_sample, loss_dict=loss_dict, outputs=None)

        return loss_dict

    def run_validation(self) -> None:
        """
        Run validation.
        """
        self.before_validation()

        self.val_dataloader_iter = iter(self.val_dataloader)

        with inference_context(self.model_parts[0]):
            with torch.no_grad():
                for step_idx in range(self.val_steps_per_epoch):
                    self.run_validation_step(step_idx)

        metrics = self.evaluator.evaluate()
        self.event_recorder.put_scalars(
            scope="epoch",
            prefix="val_",
            **{
                k: (v.item() if torch.is_tensor(v) else v)
                for k, v in metrics.items()
            },
        )
        self.evaluator.reset()

        self.after_validation()

    def run_validation_step(self, idx: int) -> None:
        """
        Run one validation step.
        """
        self.before_val_step()

        microbatch_loss_dicts = []
        for _microbatch in range(self.gradient_accumulation_steps):
            t_start = time.perf_counter()
            data_sample = next(self.val_dataloader_iter)
            data_time = time.perf_counter() - t_start
            data_sample = self.preprocessor(
                data_sample=data_sample,
                data_time=data_time,
            )

            loss_dict, outputs = self.validation_forward_step(data_sample)
            microbatch_loss_dicts.append({
                k: (v.detach() if torch.is_tensor(v) else v)
                for k, v in loss_dict.items()
            })

            self.evaluator.process(data_sample, outputs, loss_dict)

        metrics, aggregated_loss = self.metrics_processor.process(
            data_sample=data_sample,
            loss_dicts=microbatch_loss_dicts,
            extra_metrics=None
        )
        data_sample["advanced_metrics"] = metrics

        self.after_val_step(
            data_sample=data_sample,
            outputs=outputs,
            loss_dict=aggregated_loss,
        )

        self._val_iter += 1

    def validation_forward_step(self, data_sample: Sequence[dict]):
        """
        Run microbatch for validation.
        """
        if self.parallel_dims.cp_enabled:
            cp_buffers, cp_seq_dims = get_cp_buffers(data_sample, 
                                                     self.model_parts, 
                                                     disable_load_balance=True)
            optional_context_parallel_ctx = (
                dist_utils.create_context_parallel_ctx(
                    cp_mesh=self.parallel_dims.world_mesh["cp"],
                    cp_buffers=cp_buffers,
                    cp_seq_dims=cp_seq_dims,
                    cp_no_restore_buffers={cp_buffers[0], cp_buffers[1]}, # inputs, targets
                    cp_rotate_method=self.cfg.parallelism.training.context_parallel_rotate_method,
                )
            )
        else:
            optional_context_parallel_ctx = None

        if self.parallel_dims.pp_enabled:
            raise NotImplementedError(
                "Pipeline parallelism is not yet supported for validation."
            )
        else:
            with self.train_context(optional_context_parallel_ctx):
                assert len(self.model_parts) == 1
                with self.maybe_enable_amp:
                    loss_dict, outputs = self.model_parts[0](data_sample)

        return loss_dict, outputs