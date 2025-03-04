import logging
logger = logging.getLogger(__name__)

import pytest
import shutil

import warnings
warnings.filterwarnings("ignore")


from training.backend import summarize_model
from models.convnext import ConvNeXtV2

@pytest.mark.run(order=1)
def test_convnext_custom(kargs):

    # clean out existing model
    outdir = kargs['outdir']/"pytests/convnext/custom"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    logger.info(f"Testing input shape: {kargs['input_shape']}")
    inputs = (1, 6, 64, 64, 1)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = ConvNeXtV2(
        model_template='convnext',
        input_shape=inputs,
        modes=kargs['modes'],
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
    )

    summarize_model(
        model=model,
        inputs=inputs,
        batch_size=kargs['batch_size'],
        logdir=logdir,
    )


@pytest.mark.run(order=2)
def test_convnext_tiny(kargs):

    # clean out existing model
    outdir = kargs['outdir']/"pytests/convnext/tiny"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    logger.info(f"Testing input shape: {kargs['input_shape']}")
    inputs = (1, 6, 64, 64, 1)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = ConvNeXtV2(
        model_template='convnext-tiny',
        input_shape=inputs,
        modes=kargs['modes'],
    )

    summarize_model(
        model=model,
        inputs=inputs,
        batch_size=kargs['batch_size'],
        logdir=logdir,
    )


@pytest.mark.run(order=3)
def test_convnext_small(kargs):

    # clean out existing model
    outdir = kargs['outdir']/"pytests/convnext/small"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    logger.info(f"Testing input shape: {kargs['input_shape']}")
    inputs = (1, 6, 64, 64, 1)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = ConvNeXtV2(
        model_template='convnext-small',
        input_shape=inputs,
        modes=kargs['modes'],
    )

    summarize_model(
        model=model,
        inputs=inputs,
        batch_size=kargs['batch_size'],
        logdir=logdir,
    )


@pytest.mark.run(order=4)
def test_convnext_base(kargs):

    # clean out existing model
    outdir = kargs['outdir']/"pytests/convnext/base"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    logger.info(f"Testing input shape: {kargs['input_shape']}")
    inputs = (1, 6, 64, 64, 1)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = ConvNeXtV2(
        model_template='convnext-base',
        input_shape=inputs,
        modes=kargs['modes'],
    )

    summarize_model(
        model=model,
        inputs=inputs,
        batch_size=kargs['batch_size'],
        logdir=logdir,
    )


@pytest.mark.run(order=5)
def test_convnext_large(kargs):

    # clean out existing model
    outdir = kargs['outdir']/"pytests/convnext/large"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    logger.info(f"Testing input shape: {kargs['input_shape']}")
    inputs = (1, 6, 64, 64, 1)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = ConvNeXtV2(
        model_template='convnext-large',
        input_shape=inputs,
        modes=kargs['modes'],
    )

    summarize_model(
        model=model,
        inputs=inputs,
        batch_size=kargs['batch_size'],
        logdir=logdir,
    )
