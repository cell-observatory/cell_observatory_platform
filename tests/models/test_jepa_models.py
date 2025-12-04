import logging
logger = logging.getLogger(__name__)

import pytest
import shutil

import warnings
warnings.filterwarnings("ignore")

from cell_observatory_platform.models.jepa import JEPA
from cell_observatory_platform.tests.conftest import models_kargs
from cell_observatory_platform.training.helpers import summarize_model, get_masked_input_data


def test_jepa_custom(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"tests/jepa/custom"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = JEPA(
        model_template='jepa',
        input_fmt='TZYXC',
        embed_dim=models_kargs['hidden_size'],
        input_shape=inputs[1:],
        patch_shape=(1, 1, models_kargs['patches'], models_kargs['patches']),
        num_heads=models_kargs['heads'],
        depth=models_kargs['repeats'],
        modes=models_kargs['modes'],
        abs_sincos_enc=models_kargs['abs_sincos_enc'],
        rope_pos_enc=models_kargs['rope_pos_enc'],
        proj_drop_rate=models_kargs['dropout'],
        fixed_dropout_depth=models_kargs['fixed_dropout_depth'],
    ).to('cuda')

    input_data = get_masked_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )



def test_jepa_tiny(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"tests/jepa/tiny"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)

    model = JEPA(
        model_template='jepa-tiny',
        input_fmt='TZYXC',
        input_shape=inputs[1:],
        patch_shape=(1, 1, models_kargs['patches'], models_kargs['patches']),
        proj_drop_rate=models_kargs['dropout'],
        abs_sincos_enc=models_kargs['abs_sincos_enc'],
        rope_pos_enc=models_kargs['rope_pos_enc'],
        fixed_dropout_depth=models_kargs['fixed_dropout_depth'],
    ).to('cuda')

    input_data = get_masked_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )



def test_jepa_small(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"tests/jepa/small"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)

    model = JEPA(
        model_template='jepa-small',
        input_fmt='TZYXC',
        input_shape=inputs[1:],
        patch_shape=(1, 1, models_kargs['patches'], models_kargs['patches']),
        proj_drop_rate=models_kargs['dropout'],
        abs_sincos_enc=models_kargs['abs_sincos_enc'],
        rope_pos_enc=models_kargs['rope_pos_enc'],
        fixed_dropout_depth=models_kargs['fixed_dropout_depth'],
    ).to('cuda')

    input_data = get_masked_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )



def test_jepa_base(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"tests/jepa/base"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)

    model = JEPA(
        model_template='jepa-base',
        input_fmt='TZYXC',
        input_shape=inputs[1:],
        patch_shape=(1, 1, models_kargs['patches'], models_kargs['patches']),
        proj_drop_rate=models_kargs['dropout'],
        abs_sincos_enc=models_kargs['abs_sincos_enc'],
        rope_pos_enc=models_kargs['rope_pos_enc'],
        fixed_dropout_depth=models_kargs['fixed_dropout_depth'],
    ).to('cuda')

    input_data = get_masked_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )



def test_jepa_large(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"tests/jepa/large"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)

    model = JEPA(
        model_template='jepa-large',
        input_fmt='TZYXC',
        input_shape=inputs[1:],
        patch_shape=(1, 1, models_kargs['patches'], models_kargs['patches']),
        proj_drop_rate=models_kargs['dropout'],
        abs_sincos_enc=models_kargs['abs_sincos_enc'],
        rope_pos_enc=models_kargs['rope_pos_enc'],
        fixed_dropout_depth=models_kargs['fixed_dropout_depth'],
    ).to('cuda')

    input_data = get_masked_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )



def test_jepa_huge(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"tests/jepa/huge"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)

    model = JEPA(
        model_template='jepa-huge',
        input_fmt='TZYXC',
        input_shape=inputs[1:],
        patch_shape=(1, 1, models_kargs['patches'], models_kargs['patches']),
        proj_drop_rate=models_kargs['dropout'],
        abs_sincos_enc=models_kargs['abs_sincos_enc'],
        rope_pos_enc=models_kargs['rope_pos_enc'],
        fixed_dropout_depth=models_kargs['fixed_dropout_depth'],
    ).to('cuda')

    input_data = get_masked_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )