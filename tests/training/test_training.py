import logging
logger = logging.getLogger(__name__)

import pytest
import shutil

from ray import init, shutdown, cluster_resources

import warnings
warnings.filterwarnings("ignore")


from training import train


def start_ray_cluster(kargs):
    logger.info(f"Starting a new local ray cluster")
    init(
        log_to_driver=True,
        runtime_env={"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "GRAPH", "NCCL_P2P_LEVEL": "NVL"},
        num_cpus=kargs['cpu_workers'],
        num_gpus=kargs['gpu_workers'],
        ignore_reinit_error=True
    )

    logger.info('\nResources available to this Ray cluster:')
    for resource, count in cluster_resources().items():
        logger.info(f'{resource}: {count}')


# @pytest.mark.run(order=1)
# def test_supervised_training(kargs):
#
#     # clean out existing model
#     outdir = kargs['outdir']/"pytests/supervised_model"
#     if outdir.exists() and outdir.is_dir():
#         shutil.rmtree(outdir)
#
#     train.train_model(
#         network='baseline-base',
#         dataset=kargs['dataset'],
#         outdir=outdir,
#         input_shape=kargs['input_shape'],
#         modes=kargs['modes'],
#         batch_size=kargs['batch_size'],
#         hidden_size=kargs['hidden_size'],
#         patches=kargs['patches'],
#         heads=kargs['heads'],
#         repeats=kargs['repeats'],
#         opt=kargs['opt'],
#         lr=kargs['lr'],
#         wd=kargs['wd'],
#         ld=kargs['ld'],
#         ema=kargs['ema'],
#         warmup=kargs['warmup'],
#         cooldown=kargs['cooldown'],
#         clip_grad=kargs['clip_grad'],
#         epochs=kargs['epochs'],
#         fixedlr=kargs['fixedlr'],
#         dropout=kargs['dropout'],
#         fixed_dropout_depth=kargs['fixed_dropout_depth'],
#         amp=kargs['amp'],
#         finetune=kargs['finetune'],
#         profile=kargs['profile'],
#         workers=kargs['workers'],
#         gpu_workers=kargs['gpu_workers'],
#         cpu_workers=kargs['cpu_workers'],
#     )


# @pytest.mark.run(order=2)
# def test_mae_training(kargs):
#
#     # clean out existing model
#     outdir = kargs['outdir']/"pytests/mae_model"
#     if outdir.exists() and outdir.is_dir():
#         shutil.rmtree(outdir)
#
#     train.train_model(
#         network='mae-base',
#         dataset=kargs['dataset'],
#         outdir=outdir,
#         input_shape=kargs['input_shape'],
#         modes=kargs['modes'],
#         batch_size=kargs['batch_size'],
#         hidden_size=kargs['hidden_size'],
#         patches=kargs['patches'],
#         heads=kargs['heads'],
#         repeats=kargs['repeats'],
#         opt=kargs['opt'],
#         lr=kargs['lr'],
#         wd=kargs['wd'],
#         ld=kargs['ld'],
#         ema=kargs['ema'],
#         warmup=kargs['warmup'],
#         cooldown=kargs['cooldown'],
#         clip_grad=kargs['clip_grad'],
#         epochs=kargs['epochs'],
#         fixedlr=kargs['fixedlr'],
#         dropout=kargs['dropout'],
#         fixed_dropout_depth=kargs['fixed_dropout_depth'],
#         amp=kargs['amp'],
#         finetune=kargs['finetune'],
#         profile=kargs['profile'],
#         workers=kargs['workers'],
#         gpu_workers=kargs['gpu_workers'],
#         cpu_workers=kargs['cpu_workers'],
#         mae=True
#     )


# @pytest.mark.run(order=3)
# def test_jepa_training(kargs):
#
#     # clean out existing model
#     outdir = kargs['outdir']/"pytests/jepa_model"
#     if outdir.exists() and outdir.is_dir():
#         shutil.rmtree(outdir)
#
#     train.train_model(
#         network='jepa-base',
#         dataset=kargs['dataset'],
#         outdir=outdir,
#         input_shape=kargs['input_shape'],
#         modes=kargs['modes'],
#         batch_size=kargs['batch_size'],
#         hidden_size=kargs['hidden_size'],
#         patches=kargs['patches'],
#         heads=kargs['heads'],
#         repeats=kargs['repeats'],
#         opt=kargs['opt'],
#         lr=kargs['lr'],
#         wd=kargs['wd'],
#         ld=kargs['ld'],
#         ema=kargs['ema'],
#         warmup=kargs['warmup'],
#         cooldown=kargs['cooldown'],
#         clip_grad=kargs['clip_grad'],
#         epochs=kargs['epochs'],
#         fixedlr=kargs['fixedlr'],
#         dropout=kargs['dropout'],
#         fixed_dropout_depth=kargs['fixed_dropout_depth'],
#         amp=kargs['amp'],
#         finetune=kargs['finetune'],
#         profile=kargs['profile'],
#         workers=kargs['workers'],
#         gpu_workers=kargs['gpu_workers'],
#         cpu_workers=kargs['cpu_workers'],
#         jepa=True
#     )


def shutdown_ray_cluster(kargs):
    logger.info(f"Shutting down the local ray cluster")
    shutdown()
