import logging
logger = logging.getLogger(__name__)

import pytest
import shutil

import warnings
warnings.filterwarnings("ignore")

from training.helpers import summarize_model
from models.jepa import JEPA
from tests.conftest import models_kargs
from tests.helpers import get_masked_input_data


def test_jepa_custom(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"pytests/jepa/custom"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = JEPA(
        model_template='jepa',
        input_shape=inputs,
        embed_dim=models_kargs['hidden_size'],
        lateral_patch_size=models_kargs['patches'],
        axial_patch_size=1,
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



def test_jepa_tiny(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"pytests/jepa/tiny"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)

    model = JEPA(
        model_template='jepa-tiny',
        input_shape=inputs,
        lateral_patch_size=models_kargs['patches'],
        axial_patch_size=1,
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



def test_jepa_small(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"pytests/jepa/small"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)

    model = JEPA(
        model_template='jepa-small',
        input_shape=inputs,
        lateral_patch_size=models_kargs['patches'],
        axial_patch_size=1,
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



def test_jepa_base(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"pytests/jepa/base"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)

    model = JEPA(
        model_template='jepa-base',
        input_shape=inputs,
        lateral_patch_size=models_kargs['patches'],
        axial_patch_size=1,
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



def test_jepa_large(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"pytests/jepa/large"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)

    model = JEPA(
        model_template='jepa-large',
        input_shape=inputs,
        lateral_patch_size=models_kargs['patches'],
        axial_patch_size=1,
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



def test_jepa_huge(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"pytests/jepa/huge"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)

    model = JEPA(
        model_template='jepa-huge',
        input_shape=inputs,
        lateral_patch_size=models_kargs['patches'],
        axial_patch_size=1,
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
