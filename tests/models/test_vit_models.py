import logging
logger = logging.getLogger(__name__)

import pytest
import shutil

import warnings
warnings.filterwarnings("ignore")


from training.backend import summarize_model
from models.vit import ViT

@pytest.mark.run(order=1)
def test_vit_custom(kargs):

    # clean out existing model
    outdir = kargs['outdir']/"pytests/vit/custom"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    logger.info(f"Testing input shape: {kargs['input_shape']}")
    inputs = (1, 6, 64, 64, 1)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = ViT(
        model_template='vit',
        input_shape=inputs,
        embed_dim=kargs['hidden_size'],
        lateral_patch_size=kargs['patches'],
        axial_patch_size=1,
        num_heads=kargs['heads'],
        depth=kargs['repeats'],
        modes=kargs['modes'],
        proj_drop_rate=kargs['dropout'],
        fixed_dropout_depth=kargs['fixed_dropout_depth'],
    )

    summarize_model(
        model=model,
        inputs=inputs,
        batch_size=kargs['batch_size'],
        logdir=logdir,
    )


@pytest.mark.run(order=2)
def test_vit_tiny(kargs):

    # clean out existing model
    outdir = kargs['outdir']/"pytests/vit/tiny"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    logger.info(f"Testing input shape: {kargs['input_shape']}")
    inputs = (1, 6, 64, 64, 1)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = ViT(
        model_template='vit-tiny',
        input_shape=inputs,
        lateral_patch_size=kargs['patches'],
        axial_patch_size=1,
        proj_drop_rate=kargs['dropout'],
        fixed_dropout_depth=kargs['fixed_dropout_depth'],
    )

    summarize_model(
        model=model,
        inputs=inputs,
        batch_size=kargs['batch_size'],
        logdir=logdir,
    )


@pytest.mark.run(order=3)
def test_vit_small(kargs):

    # clean out existing model
    outdir = kargs['outdir']/"pytests/vit/small"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    logger.info(f"Testing input shape: {kargs['input_shape']}")
    inputs = (1, 6, 64, 64, 1)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = ViT(
        model_template='vit-small',
        input_shape=inputs,
        lateral_patch_size=kargs['patches'],
        axial_patch_size=1,
        proj_drop_rate=kargs['dropout'],
        fixed_dropout_depth=kargs['fixed_dropout_depth'],
    )

    summarize_model(
        model=model,
        inputs=inputs,
        batch_size=kargs['batch_size'],
        logdir=logdir,
    )


@pytest.mark.run(order=4)
def test_vit_base(kargs):

    # clean out existing model
    outdir = kargs['outdir']/"pytests/vit/base"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    logger.info(f"Testing input shape: {kargs['input_shape']}")
    inputs = (1, 6, 64, 64, 1)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = ViT(
        model_template='vit-base',
        input_shape=inputs,
        lateral_patch_size=kargs['patches'],
        axial_patch_size=1,
        proj_drop_rate=kargs['dropout'],
        fixed_dropout_depth=kargs['fixed_dropout_depth'],
    )

    summarize_model(
        model=model,
        inputs=inputs,
        batch_size=kargs['batch_size'],
        logdir=logdir,
    )


@pytest.mark.run(order=5)
def test_vit_large(kargs):

    # clean out existing model
    outdir = kargs['outdir']/"pytests/vit/large"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    logger.info(f"Testing input shape: {kargs['input_shape']}")
    inputs = (1, 6, 64, 64, 1)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = ViT(
        model_template='vit-large',
        input_shape=inputs,
        lateral_patch_size=kargs['patches'],
        axial_patch_size=1,
        proj_drop_rate=kargs['dropout'],
        fixed_dropout_depth=kargs['fixed_dropout_depth'],
    )

    summarize_model(
        model=model,
        inputs=inputs,
        batch_size=kargs['batch_size'],
        logdir=logdir,
    )


@pytest.mark.run(order=6)
def test_vit_huge(kargs):

    # clean out existing model
    outdir = kargs['outdir']/"pytests/vit/huge"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    logger.info(f"Testing input shape: {kargs['input_shape']}")
    inputs = (1, 6, 64, 64, 1)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = ViT(
        model_template='vit-huge',
        input_shape=inputs,
        lateral_patch_size=kargs['patches'],
        axial_patch_size=1,
        proj_drop_rate=kargs['dropout'],
        fixed_dropout_depth=kargs['fixed_dropout_depth'],
    )

    summarize_model(
        model=model,
        inputs=inputs,
        batch_size=kargs['batch_size'],
        logdir=logdir,
    )
