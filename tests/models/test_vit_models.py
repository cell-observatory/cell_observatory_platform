import logging
logger = logging.getLogger(__name__)

import pytest
import shutil

import warnings
warnings.filterwarnings("ignore")

from training.helpers import summarize_model
from models.vit import ViT
from tests.helpers import get_input_data


def test_vit_custom(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"pytests/vit/custom"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = ViT(
        model_template='vit',
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

    input_data = get_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )



def test_vit_tiny(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"pytests/vit/tiny"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = ViT(
        model_template='vit-tiny',
        input_shape=inputs,
        lateral_patch_size=models_kargs['patches'],
        axial_patch_size=1,
        proj_drop_rate=models_kargs['dropout'],
        fixed_dropout_depth=models_kargs['fixed_dropout_depth'],
    )

    input_data = get_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )



def test_vit_small(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"pytests/vit/small"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = ViT(
        model_template='vit-small',
        input_shape=inputs,
        lateral_patch_size=models_kargs['patches'],
        axial_patch_size=1,
        proj_drop_rate=models_kargs['dropout'],
        fixed_dropout_depth=models_kargs['fixed_dropout_depth'],
    )

    input_data = get_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )


def test_vit_base(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"pytests/vit/base"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = ViT(
        model_template='vit-base',
        input_shape=inputs,
        lateral_patch_size=models_kargs['patches'],
        axial_patch_size=1,
        proj_drop_rate=models_kargs['dropout'],
        fixed_dropout_depth=models_kargs['fixed_dropout_depth'],
    )

    input_data = get_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )



def test_vit_large(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"pytests/vit/large"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = ViT(
        model_template='vit-large',
        input_shape=inputs,
        lateral_patch_size=models_kargs['patches'],
        axial_patch_size=1,
        proj_drop_rate=models_kargs['dropout'],
        fixed_dropout_depth=models_kargs['fixed_dropout_depth'],
    )

    input_data = get_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )


def test_vit_huge(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"pytests/vit/huge"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = ViT(
        model_template='vit-huge',
        input_shape=inputs,
        lateral_patch_size=models_kargs['patches'],
        axial_patch_size=1,
        proj_drop_rate=models_kargs['dropout'],
        fixed_dropout_depth=models_kargs['fixed_dropout_depth'],
    )

    input_data = get_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )