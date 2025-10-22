import logging
logger = logging.getLogger(__name__)

import pytest
import shutil

import warnings
warnings.filterwarnings("ignore")


from training.helpers import summarize_model, get_input_data
from models.baseline import Baseline
from tests.conftest import models_kargs


def test_baseline_custom(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"tests/baseline/custom"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = Baseline(
        model_template='baseline',
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
        abs_sincos_enc=models_kargs['abs_sincos_enc'],
        rope_pos_enc=models_kargs['rope_pos_enc'],
    ).to('cuda')

    input_data = get_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )



def test_baseline_tiny(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"tests/baseline/tiny"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = Baseline(
        model_template='baseline-tiny',
        input_fmt='TZYXC',
        input_shape=inputs,
        lateral_patch_size=models_kargs['patches'],
        axial_patch_size=1,
        temporal_patch_size=1,
        proj_drop_rate=models_kargs['dropout'],
        fixed_dropout_depth=models_kargs['fixed_dropout_depth'],
        abs_sincos_enc=models_kargs['abs_sincos_enc'],
        rope_pos_enc=models_kargs['rope_pos_enc'],
    ).to('cuda')

    input_data = get_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )



def test_baseline_small(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"tests/baseline/small"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = Baseline(
        model_template='baseline-small',
        input_fmt='TZYXC',
        input_shape=inputs,
        lateral_patch_size=models_kargs['patches'],
        axial_patch_size=1,
        temporal_patch_size=1,
        proj_drop_rate=models_kargs['dropout'],
        fixed_dropout_depth=models_kargs['fixed_dropout_depth'],
        abs_sincos_enc=models_kargs['abs_sincos_enc'],
        rope_pos_enc=models_kargs['rope_pos_enc'],
    ).to('cuda')

    input_data = get_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )



def test_baseline_base(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"tests/baseline/base"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = Baseline(
        model_template='baseline-base',
        input_fmt='TZYXC',
        input_shape=inputs,
        lateral_patch_size=models_kargs['patches'],
        axial_patch_size=1,
        temporal_patch_size=1,
        proj_drop_rate=models_kargs['dropout'],
        fixed_dropout_depth=models_kargs['fixed_dropout_depth'],
        abs_sincos_enc=models_kargs['abs_sincos_enc'],
        rope_pos_enc=models_kargs['rope_pos_enc'],
    ).to('cuda')

    input_data = get_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )



def test_baseline_large(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"tests/baseline/large"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = Baseline(
        model_template='baseline-large',
        input_fmt='TZYXC',
        input_shape=inputs,
        lateral_patch_size=models_kargs['patches'],
        axial_patch_size=1,
        temporal_patch_size=1,
        proj_drop_rate=models_kargs['dropout'],
        fixed_dropout_depth=models_kargs['fixed_dropout_depth'],
        abs_sincos_enc=models_kargs['abs_sincos_enc'],
        rope_pos_enc=models_kargs['rope_pos_enc'],
    ).to('cuda')

    input_data = get_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )



def test_baseline_huge(models_kargs):

    # clean out existing model
    outdir = models_kargs['outdir']/"tests/baseline/huge"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)


    model = Baseline(
        model_template='baseline-huge',
        input_fmt='TZYXC',
        input_shape=inputs,
        lateral_patch_size=models_kargs['patches'],
        axial_patch_size=1,
        temporal_patch_size=1,
        proj_drop_rate=models_kargs['dropout'],
        fixed_dropout_depth=models_kargs['fixed_dropout_depth'],
        abs_sincos_enc=models_kargs['abs_sincos_enc'],
        rope_pos_enc=models_kargs['rope_pos_enc'],
    ).to('cuda')

    input_data = get_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs['batch_size'],
        logdir=logdir,
    )
