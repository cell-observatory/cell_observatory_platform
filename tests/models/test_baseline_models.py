import logging
logger = logging.getLogger(__name__)

import pytest
import shutil

import warnings
warnings.filterwarnings("ignore")


from training.backend import summarize_model
from models.baseline import Baseline

@pytest.mark.run(order=1)
def test_baseline_custom(kargs):

    # clean out existing model
    outdir = kargs['outdir']/"pytests/baseline/custom"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    logger.info(f"Testing input shape: {kargs['input_shape']}")
    inputs = (1, 6, 64, 64, 1)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = Baseline(
        model_template='baseline',
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
def test_baseline_tiny(kargs):

    # clean out existing model
    outdir = kargs['outdir']/"pytests/baseline/tiny"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    logger.info(f"Testing input shape: {kargs['input_shape']}")
    inputs = (1, 6, 64, 64, 1)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = Baseline(
        model_template='baseline-tiny',
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
def test_baseline_small(kargs):

    # clean out existing model
    outdir = kargs['outdir']/"pytests/baseline/small"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    logger.info(f"Testing input shape: {kargs['input_shape']}")
    inputs = (1, 6, 64, 64, 1)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = Baseline(
        model_template='baseline-small',
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
def test_baseline_base(kargs):

    # clean out existing model
    outdir = kargs['outdir']/"pytests/baseline/base"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    logger.info(f"Testing input shape: {kargs['input_shape']}")
    inputs = (1, 6, 64, 64, 1)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = Baseline(
        model_template='baseline-base',
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
def test_baseline_large(kargs):

    # clean out existing model
    outdir = kargs['outdir']/"pytests/baseline/large"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    logger.info(f"Testing input shape: {kargs['input_shape']}")
    inputs = (1, 6, 64, 64, 1)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = Baseline(
        model_template='baseline-large',
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
def test_baseline_huge(kargs):

    # clean out existing model
    outdir = kargs['outdir']/"pytests/baseline/huge"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    logger.info(f"Testing input shape: {kargs['input_shape']}")
    inputs = (1, 6, 64, 64, 1)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = Baseline(
        model_template='baseline-huge',
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
