import logging
logger = logging.getLogger(__name__)

import pytest
import shutil

import warnings
warnings.filterwarnings("ignore")

from tests.conftest import models_kargs
from training.helpers import summarize_model
from tests.helpers import get_masked_input_data
from models.maskedautoencoder import MaskedAutoEncoder


def test_mae_custom(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"pytests/mae/custom"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = MaskedAutoEncoder(
        model_template='mae',
        input_fmt='TZYXC',
        input_shape=inputs,
        embed_dim=models_kargs['hidden_size'],
        lateral_patch_size=models_kargs['patches'],
        axial_patch_size=1,
        temporal_patch_size=1,
        num_heads=models_kargs['heads'],
        depth=models_kargs['repeats'],
        modes=models_kargs['modes'],
        proj_drop_rate=models_kargs['dropout'],
        fixed_dropout_depth=models_kargs['fixed_dropout_depth'],
    )

    input_data = get_masked_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )



def test_mae_tiny(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"pytests/mae/tiny"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = MaskedAutoEncoder(
        model_template='mae-tiny',
        input_fmt='TZYXC',
        input_shape=inputs,
        lateral_patch_size=models_kargs['patches'],
        axial_patch_size=1,
        temporal_patch_size=1,
        proj_drop_rate=models_kargs['dropout'],
        fixed_dropout_depth=models_kargs['fixed_dropout_depth'],
    )

    input_data = get_masked_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )



def test_mae_small(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"pytests/mae/small"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = MaskedAutoEncoder(
        model_template='mae-small',
        input_fmt='TZYXC',
        input_shape=inputs,
        lateral_patch_size=models_kargs['patches'],
        axial_patch_size=1,
        temporal_patch_size=1,
        proj_drop_rate=models_kargs['dropout'],
        fixed_dropout_depth=models_kargs['fixed_dropout_depth'],
    )

    input_data = get_masked_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )



def test_mae_base(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"pytests/mae/base"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = MaskedAutoEncoder(
        model_template='mae-base',
        input_fmt='TZYXC',
        input_shape=inputs,
        lateral_patch_size=models_kargs['patches'],
        axial_patch_size=1,
        temporal_patch_size=1,
        proj_drop_rate=models_kargs['dropout'],
        fixed_dropout_depth=models_kargs['fixed_dropout_depth'],
    )

    input_data = get_masked_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )



def test_mae_large(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"pytests/mae/large"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = MaskedAutoEncoder(
        model_template='mae-large',
        input_fmt='TZYXC',
        input_shape=inputs,
        lateral_patch_size=models_kargs['patches'],
        axial_patch_size=1,
        temporal_patch_size=1,
        proj_drop_rate=models_kargs['dropout'],
        fixed_dropout_depth=models_kargs['fixed_dropout_depth'],
    )

    input_data = get_masked_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )



def test_mae_huge(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"pytests/mae/huge"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = MaskedAutoEncoder(
        model_template='mae-huge',
        input_fmt='TZYXC',
        input_shape=inputs,
        lateral_patch_size=models_kargs['patches'],
        axial_patch_size=1,
        temporal_patch_size=1,
        proj_drop_rate=models_kargs['dropout'],
        fixed_dropout_depth=models_kargs['fixed_dropout_depth'],
    )

    input_data = get_masked_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )
