"""
Adopted with Apache License 2.0 from
https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/train_loop.py
https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
"""

import time
import logging
import weakref
from typing import List, Optional, Sequence

from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import instantiate, get_class
if not OmegaConf.has_resolver("now"):
    OmegaConf.register_new_resolver("now", lambda fmt: time.strftime(fmt))

from ray.train import get_context

import torch
from deepspeed import initialize

from training.helpers import (
    get_steps_per_epoch,
    resume_run,
    summarize_model,
    get_masked_input_data,
    enable_optimizations
)
from training.hooks import HookBase
from training.loggers import EventRecorder
from data.dataloaders import get_dataloader
from training.optimizers import get_optimizer
from training.schedulers import get_schedulers, get_param_groups
from training.registry import build_dependency_graph_and_instantiate
from utils.context import inference_context, process_rank, unlink_shared_memory

logger = logging.getLogger("ray")
logger.setLevel(logging.INFO)

# silence broken logging call in Ray internals to prevent
# checkpoint saving from failing
logging.getLogger("ray.train._internal.checkpoint_manager").setLevel(logging.INFO)


# Ray train wrapper entry point
def train_loop_per_worker(config):
    trainer_cls = get_class(config.trainer)
    trainer_per_worker = trainer_cls(config)

    if config.job_type == "train":
        trainer_per_worker.run()
    elif config.job_type == "test":
        trainer_per_worker.test()
    else:
        raise ValueError(f"Unknown job type: {config.job_type}. "
                         f"Expected 'train' or 'test', got '{config.job_type}'.")
    
    return {"best_metric": trainer_per_worker.best_metric}


class BaseTrainer:
    """
    Base class for iterative trainer with hooks.
    """

    def __init__(self, config: DictConfig) -> None:
        # initialize event recorder
        self.event_recorder: EventRecorder = instantiate(config.loggers.event_recorder)

        # initialize event_writers
        event_writers = self._build_event_writers(
            w_cfgs=config.loggers.event_writers, 
            recorder=self.event_recorder
        )
        self.event_writers_list = instantiate(
            config.loggers.event_writers_list,
            writers = event_writers
        )
        
        # intialize hooks
        hooks = self._build_hooks(config.hooks.hooks_list, self.event_writers_list)
        self._hooks: List[HookBase] = []
        self.register_hooks(hooks)

    @staticmethod
    def _build_event_writers(w_cfgs, recorder):
        writers = []
        for writer_cfg in w_cfgs:
            writer = instantiate(writer_cfg, event_recorder=recorder)
            writers.append(writer)
        return writers

    @staticmethod
    def _build_hooks(h_cfgs, event_writers):
        hooks = []
        for hc in h_cfgs:
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
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
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

    def after_backward(self):
        for h in self._hooks:
            h.after_backward()

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

        self.ray_context = get_context()
        self.val_begin, self.val_interval = cfg.evaluation.val_begin, cfg.evaluation.val_interval
        self.stop_training, self._max_epochs = False, cfg.schedulers.epochs

        # initialize dataset and dataloader
        # get_dataloader() returns a tuple of dataloaders
        # (train_dataloader, val_dataloader) where
        # val_dataloader is None if no validation set is provided
        self.train_dataloader, self.val_dataloader, \
            self.host_buffer_actor, self.device_buffer = get_dataloader(cfg)

        self.steps_per_epoch, self.val_steps_per_epoch = get_steps_per_epoch(
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            config=cfg
        )

        self.preprocessor = instantiate(cfg.datasets.preprocessor)

        # initialize model
        # TODO: consider migrating to BUILD() based initialization
        #       instead of recursive instantiation
        model = build_dependency_graph_and_instantiate(cfg.models)

        # FIXME: there seems to be a bug in model definitions where we  
        #       have the model defined in a subfolder (e.g. models/abc/model.py)
        #       this hack works for one folder deep models but should be fixed
        model, = model.values() if isinstance(model, dict) else (model,)

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
        opt, _ = get_optimizer(
            params=param_groups,
            config=cfg,
            optimizer=cfg.optimizers.opt,
            steps_per_epoch=self.steps_per_epoch
        )
        self.scheduler, self.wd_scheduler = get_schedulers(
            opt=opt,
            config=cfg,
            steps_per_epoch=self.steps_per_epoch
        )

        # enable optimizations if specified
        # includes setting Torch backend flags, 
        # activation checkpointing, and torch Compile 
        if cfg.optimizations is not None:
            model = enable_optimizations(cfg=cfg, model=model)

        # initialize deepspeed
        self.model, self.opt, _, _ = initialize(
            model=model,
            optimizer=opt,
            config=OmegaConf.to_container(cfg.deepspeed, resolve=True)
        )

        # initialize checkpoint manager
        self.checkpoint_manager = instantiate(
            cfg.checkpoint.checkpoint_manager,
            model=self.model
        )

        if cfg.checkpoint.checkpoint_manager.pretrained_checkpointdir:
            # load model state from checkpoint
            # if load_checkpointdir is not None, 
            # the model will be loaded from the specified directory.
            # this is different from loading a checkpoint
            # in resume_run() where a pre-existing checkpoint is  
            # loaded to resume a job. here we load a checkpoint to
            # start a new job with part of or the entire state of
            # a pre-existing model (e.g. for fine-tuning) instead of
            # for resuming a job that failed due to training instability.
            # should only be used with resume_run=False.
            self.checkpoint_manager.load()

        # if resume job, gather the state from the checkpoint
        # else intialize outdir, logdir, and checkpointdir
        # these directories must be empty if not resuming a job
        # to avoid overwriting existing checkpoints
        # see training/utils.py:resume_run()
        # and training/run.py
        best_metric, step, epoch = resume_run(self, cfg)
        self.start_epoch, self.start_iter, self.best_metric = epoch, step, best_metric
        self._epoch, self._iter, self._val_iter = self.start_epoch, self.start_iter, 0

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
        
        end = time.perf_counter()
        for idx, data_sample in enumerate(self.train_dataloader):
            data_time = time.perf_counter() - end
            data_sample = self.preprocessor(data_sample=data_sample, data_time=data_time)
            # run one step with the fetched data sample
            self.run_step(idx, data_sample)
            end = time.perf_counter()

        if self.val_dataloader and \
           (self._epoch >= self.val_begin and
            (self._epoch - self.val_begin) % self.val_interval == 0):
            # run validation
            self.run_validation()

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
        loss_dict, outputs = self.model(data_sample)
        self.model.backward(loss_dict["step_loss"])
        self.model.step()

        # for short testing runs:
        # if idx > 25:
        #     raise RuntimeError(
        #         f"Training stopped at step {idx} for testing."
        #     )
        # logger.info(f"step_loss: {loss_dict['step_loss']}, lr: {self.opt.param_groups[0]['lr']}")

        self.after_step(data_sample=data_sample, outputs=outputs, loss_dict=loss_dict)
        self._iter += 1

    def run_validation(self) -> None:
        """
        Run validation.
        """
        self.before_validation()
        # technically, contexts could be a hook
        # but kept here for clarity
        with inference_context(self.model):
            with torch.no_grad():
                end = time.perf_counter()
                for idx, data_sample in enumerate(self.val_dataloader):
                    data_time = time.perf_counter() - end
                    data_sample = self.preprocessor(data_sample=data_sample, data_time=data_time)
                    # run one step with the fetched data sample
                    self.run_validation_step(idx, data_sample)
                    end = time.perf_counter()

        metrics = self.evaluator.evaluate()
        self.event_recorder.put_scalars(
            scope="epoch",
            prefix="val_",
            **{k: (v.item() if torch.is_tensor(v) else v)
                for k, v in metrics.items()
            }
        )
        self.evaluator.reset()

        self.after_validation()
    
    def run_validation_step(self, idx: int, data_sample: Sequence[dict]) -> None:
        """
        Iterate one validation step.
        """
        self.before_val_step()

        loss_dict, outputs = self.model(data_sample)
        self.evaluator.process(data_sample, outputs, loss_dict)

        self.after_val_step(data_sample=data_sample, outputs=outputs, loss_dict=loss_dict)
        self._val_iter += 1


class TestTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        
        self.ray_context = get_context()
        self.event_recorder._iter = 0
        self.event_recorder._epoch = 0
        self._iter, self.start_iter, self.start_epoch = 0, 0, 0

        # initialize dataset and dataloader
        self.test_dataloader, _ = get_dataloader(cfg)

        self.steps_per_epoch, val_steps_per_epoch = get_steps_per_epoch(
            train_dataloader=self.test_dataloader,
            val_dataloader=None,
            config=cfg
        )

        self.preprocessor = instantiate(cfg.datasets.preprocessor)

        # initialize model
        model = build_dependency_graph_and_instantiate(cfg.models)

        # initialize optimizer
        opt, _ = get_optimizer(
            params=model.parameters(),
            config=cfg,
            optimizer=cfg.optimizers.opt,
            steps_per_epoch=self.steps_per_epoch
        )

        # initialize deepspeed
        self.model, self.opt, _, _ = initialize(
            model=model,
            optimizer=opt,
            config=OmegaConf.to_container(cfg.deepspeed, resolve=True)
        )

        # initialize checkpoint manager and
        # load model state from checkpoint
        self.checkpoint_manager = instantiate(
            cfg.checkpoint.checkpoint_manager,
            model=self.model
        )
        self.checkpoint_manager.load()

        # initialize evaluator
        self.evaluator = instantiate(cfg.evaluation.evaluator)

    def test(self):
        """
        Run Model testing.
        """
        self.before_test()

        with inference_context(self.model):
            with torch.no_grad():
                for idx, data_sample in enumerate(self.test_dataloader):
                    self.run_test_step(idx, data_sample)
        
        metrics = self.evaluator.evaluate()
        self.event_recorder.put_scalars(
            prefix="test_",
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

        data_sample = self.preprocessor(data_sample)
        loss_dict, outputs = self.model(data_sample)
        self.evaluator.process(data_sample, outputs, loss_dict)

        self.after_test_step(data_sample=data_sample, 
                             outputs=outputs, loss_dict=loss_dict)
        self._iter += 1