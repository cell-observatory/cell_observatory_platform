import os
import sys
import pytest
import logging
from pathlib import Path

from dotenv import load_dotenv
from hydra.utils import get_class
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf, open_dict
OmegaConf.register_new_resolver("eval", eval)

import torch
from ray.train import report
from ray import init, cluster_resources
from ray.train.torch import TorchTrainer, TorchConfig
from ray.train import ScalingConfig, CheckpointConfig, RunConfig, FailureConfig

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# TODO: we should probably have separate configs
#       for testing modules
def make_config() -> DictConfig:
    with initialize(config_path="../../configs"):
        return compose(config_name="pretrain_mae_local")

def _test_ckpt_dist(cfg: DictConfig):    
    init(log_to_driver=True,
         runtime_env={k: v for k, v in os.environ.items()},
         num_cpus=cfg.clusters.total_cpus + cfg.clusters.cpus_for_training_coordinator,
         num_gpus=cfg.clusters.total_gpus,
         ignore_reinit_error=True
    )
    
    for resource, count in cluster_resources().items():
        logger.info(f'{resource}: {count}')
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    scaling_config = ScalingConfig(
        num_workers=cfg.clusters.scaling_config.num_workers,
        resources_per_worker=cfg.clusters.scaling_config.resources_per_worker,
        trainer_resources=cfg.clusters.scaling_config.trainer_resources,
        use_gpu=cfg.clusters.scaling_config.use_gpu
    )

    checkpoint_config = CheckpointConfig(**cfg.checkpoint.ray_checkpoint_config)
    run_config = RunConfig(
        log_to_file=cfg.clusters.run_config.log_to_file,
        checkpoint_config=checkpoint_config,
        failure_config=FailureConfig(max_failures=0),
        storage_path=cfg.clusters.run_config.storage_path,
    )
    
    torch_config = TorchConfig(timeout_s=cfg.clusters.torch_config.timeout_s)

    # train function to use, this should stay inside
    # the setup_ray_cluster function to prevent serialization issues
    def run_init(config: DictConfig):
        from tests.conftest import get_masked_input_data
        trainer_cls = get_class(config.trainer)
        trainer_per_worker = trainer_cls(config)

        # it is tempting to set batches lower
        # but if batches is too low checkpoint saving
        # does not work correctly probably due to the fact that
        # the sharding of the model is not done correctly/completely
        def dummy_loader(model, batches=100):
            inputs = [1] + config.datasets.input_shape
            for _ in range(batches):
                yield get_masked_input_data(model, inputs)

        def _move_to_device(batch, device):
            if isinstance(batch, (list, tuple)):
                return [_move_to_device(b, device=device) for b in batch]
            elif isinstance(batch, dict):
                return {k: _move_to_device(v, device=device) for k, v in batch.items()}
            elif isinstance(batch, torch.Tensor):
                return batch.to(device, non_blocking=True)
            else:
                return batch

        # we need to do a few steps for 
        # some state dict fields to be populated 
        # such as exp_avg etc.
        model = trainer_per_worker.model
        for batch in dummy_loader(model):
            batch = _move_to_device(batch, device=model.device)
            loss_dict, outputs = model(*batch)
            model.backward(loss_dict["step_loss"])
            model.step()
        
        # in the save test we save a checkpoint and 
        # record start_epoch, start_iter, and best_metric
        # with dummy value of 42 and in load checkpoint test
        # we load the same checkpoints for all combinations of zero stages
        # and check if the client state is restored correctly, i.e. if 
        # start_epoch, start_iter, and best_metric are all 42
        # TODO: add check to ensure weights are identical
        if config.save_checkpoint:
            trainer_per_worker.checkpoint_manager.save(
                prefix=config.checkpoint.checkpoint_manager.checkpoint_tag, 
                save_epoch=42,
                save_best_loss=42,
                save_step=42
            )
            metrics = {"success": True}
        else:
            metrics = {
                "success": True,
                "start_epoch": trainer_per_worker.start_epoch,
                "start_iter": trainer_per_worker.start_iter,
                "best_metric": trainer_per_worker.best_metric,
            }

        report(metrics) 

    trainer = TorchTrainer(
        train_loop_per_worker=run_init,
        train_loop_config=cfg,
        run_config=run_config,
        scaling_config=scaling_config,
        torch_config=torch_config,
        datasets=None
    )

    result = trainer.fit()
    return {
        "success": result.metrics.get("success", False),
        "best_metric": result.metrics.get("best_metric", None),
        "start_epoch": result.metrics.get("start_epoch", None),
        "start_iter": result.metrics.get("start_iter", None),
    }

@pytest.mark.order(1)
@pytest.mark.parametrize("zero_stage", [1]) # 2, 3
def test_checkpoint_save(zero_stage: int):    
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available for testing")
    else:
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("At least 2 GPUs are required for this test")

    config = make_config()
    load_dotenv(config.paths.dotenv_path, verbose=True)

    with open_dict(config):
        config.deepspeed.zero_optimization.stage = zero_stage
        config.experiment_name = "test_checkpoint"
        config.paths.resume_checkpointdir = None  # no resume checkpoint for saving test
        config.checkpoint.checkpoint_manager.checkpoint_tag = f"stage{zero_stage}"
        config.clusters.worker_nodes = 1
        config.clusters.gpus_per_worker = 2 # need >=2 for testing zero stages > 0
        config.clusters.cpus_per_gpu = 4
        config.clusters.mem_per_cpu = 31000

        config.save_checkpoint = True
    
    result_dict = _test_ckpt_dist(config)
    assert result_dict["success"], "Test did not complete successfully"

@pytest.mark.order(2)
@pytest.mark.parametrize("zero_stage_dst", [1]) # , 2, 3
@pytest.mark.parametrize("zero_stage_src", [1]) # , 2, 3
def test_checkpoint_load(zero_stage_src: int, zero_stage_dst: int):
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available for testing")
    else:
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("At least 2 GPUs are required for this test")

    config = make_config()
    load_dotenv(config.paths.dotenv_path, verbose=True)

    with open_dict(config):
        config.deepspeed.zero_optimization.stage = zero_stage_dst
        config.experiment_name = f"test_checkpoint"
        config.paths.resume_checkpointdir = os.path.join(config.paths.outdir, "checkpoints")
        config.checkpoint.checkpoint_manager.checkpoint_tag = f"stage{zero_stage_src}"
        config.clusters.worker_nodes = 1
        config.clusters.gpus_per_worker = 2 # need >=2 for testing zero stages > 0
        config.clusters.cpus_per_gpu = 4
        config.clusters.mem_per_cpu = 31000
        config.deepspeed.checkpoint.load_universal = True

        config.save_checkpoint = False
    
    result_dict = _test_ckpt_dist(config)

    assert result_dict["start_epoch"] == 42, f"Expected start_epoch to be 42, got {result_dict['start_epoch']}"
    assert result_dict["start_iter"] == 42, f"Expected start_iter to be 42, got {result_dict['start_iter']}"
    assert result_dict["best_metric"] == 42, f"Expected best_metric to be 42, got {result_dict['best_metric']}"
    assert result_dict["success"], "Test did not complete successfully"