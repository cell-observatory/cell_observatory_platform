import logging
logger = logging.getLogger(__name__)

import pytest
import shutil

import warnings
warnings.filterwarnings("ignore")


from training.helpers import summarize_model, get_input_data
from models.convnext import ConvNeXtV2
from tests.conftest import models_kargs


def test_convnext_custom(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"tests/convnext/custom"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = ConvNeXtV2(
        model_template='convnext',
        input_fmt='ZYXC',
        input_shape=inputs,
        modes=models_kargs['modes'],
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
    ).to('cuda')

    input_data = get_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )



def test_convnext_tiny(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"tests/convnext/tiny"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = ConvNeXtV2(
        model_template='convnext-tiny',
        input_fmt='ZYXC',
        input_shape=inputs,
        modes=models_kargs['modes'],
    ).to('cuda')

    input_data = get_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )



def test_convnext_small(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"tests/convnext/small"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = ConvNeXtV2(
        model_template='convnext-small',
        input_fmt='ZYXC',
        input_shape=inputs,
        modes=models_kargs['modes'],
    ).to('cuda')

    input_data = get_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )



def test_convnext_base(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"tests/convnext/base"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = ConvNeXtV2(
        model_template='convnext-base',
        input_fmt='ZYXC',
        input_shape=inputs,
        modes=models_kargs['modes'],
    ).to('cuda')

    input_data = get_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )


def test_convnext_large(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"tests/convnext/large"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = ConvNeXtV2(
        model_template='convnext-large',
        input_fmt='ZYXC',
        input_shape=inputs,
        modes=models_kargs['modes'],
    ).to('cuda')

    input_data = get_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )