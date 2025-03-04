import logging
logger = logging.getLogger(__name__)

import pytest
import shutil

import warnings
warnings.filterwarnings("ignore")


from training.backend import summarize_model
from models.maskedautoencoder import MaskedAutoEncoder

@pytest.mark.run(order=1)
def test_mae_custom(kargs):

    # clean out existing model
    outdir = kargs['outdir']/"pytests/mae/custom"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    logger.info(f"Testing input shape: {kargs['input_shape']}")
    inputs = (1, 6, 64, 64, 1)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = MaskedAutoEncoder(
        model_template='mae',
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
def test_mae_tiny(kargs):

    # clean out existing model
    outdir = kargs['outdir']/"pytests/mae/tiny"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    logger.info(f"Testing input shape: {kargs['input_shape']}")
    inputs = (1, 6, 64, 64, 1)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = MaskedAutoEncoder(
        model_template='mae-tiny',
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
def test_mae_small(kargs):

    # clean out existing model
    outdir = kargs['outdir']/"pytests/mae/small"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    logger.info(f"Testing input shape: {kargs['input_shape']}")
    inputs = (1, 6, 64, 64, 1)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = MaskedAutoEncoder(
        model_template='mae-small',
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
def test_mae_base(kargs):

    # clean out existing model
    outdir = kargs['outdir']/"pytests/mae/base"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    logger.info(f"Testing input shape: {kargs['input_shape']}")
    inputs = (1, 6, 64, 64, 1)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = MaskedAutoEncoder(
        model_template='mae-base',
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
def test_mae_large(kargs):

    # clean out existing model
    outdir = kargs['outdir']/"pytests/mae/large"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    logger.info(f"Testing input shape: {kargs['input_shape']}")
    inputs = (1, 6, 64, 64, 1)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = MaskedAutoEncoder(
        model_template='mae-large',
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
def test_mae_huge(kargs):

    # clean out existing model
    outdir = kargs['outdir']/"pytests/mae/huge"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    logger.info(f"Testing input shape: {kargs['input_shape']}")
    inputs = (1, 6, 64, 64, 1)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = MaskedAutoEncoder(
        model_template='mae-huge',
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
