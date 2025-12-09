import logging

logger = logging.getLogger(__name__)

import shutil
import warnings

import pytest

warnings.filterwarnings("ignore")

from cell_observatory_platform.models.meta_arch.maskedautoencoder import MaskedAutoEncoder
from cell_observatory_platform.tests.conftest import models_kargs
from cell_observatory_platform.training.helpers import get_masked_input_data, summarize_model


def test_mae_custom(models_kargs):

    # clean out existing model
    outdir = models_kargs["outdir"] / "tests/mae/custom"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / "logs"
    logdir.mkdir(exist_ok=True, parents=True)

    model = MaskedAutoEncoder(
        model_template="mae",
        input_fmt="TZYXC",
        embed_dim=models_kargs["hidden_size"],
        input_shape=inputs[1:],
        patch_shape=(1, 1, models_kargs["patches"], models_kargs["patches"]),
        num_heads=models_kargs["heads"],
        depth=models_kargs["repeats"],
        modes=models_kargs["modes"],
        proj_drop_rate=models_kargs["dropout"],
        fixed_dropout_depth=models_kargs["fixed_dropout_depth"],
        abs_sincos_enc=models_kargs["abs_sincos_enc"],
        rope_pos_enc=models_kargs["rope_pos_enc"],
    ).to("cuda")

    input_data = get_masked_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs["batch_size"],
        logdir=logdir,
    )


def test_mae_tiny(models_kargs):

    # clean out existing model
    outdir = models_kargs["outdir"] / "tests/mae/tiny"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / "logs"
    logdir.mkdir(exist_ok=True, parents=True)

    model = MaskedAutoEncoder(
        model_template="mae-tiny",
        input_fmt="TZYXC",
        input_shape=inputs[1:],
        patch_shape=(1, 1, models_kargs["patches"], models_kargs["patches"]),
        proj_drop_rate=models_kargs["dropout"],
        fixed_dropout_depth=models_kargs["fixed_dropout_depth"],
        abs_sincos_enc=models_kargs["abs_sincos_enc"],
        rope_pos_enc=models_kargs["rope_pos_enc"],
    ).to("cuda")

    input_data = get_masked_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs["batch_size"],
        logdir=logdir,
    )


def test_mae_small(models_kargs):

    # clean out existing model
    outdir = models_kargs["outdir"] / "tests/mae/small"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / "logs"
    logdir.mkdir(exist_ok=True, parents=True)

    model = MaskedAutoEncoder(
        model_template="mae-small",
        input_fmt="TZYXC",
        input_shape=inputs[1:],
        patch_shape=(1, 1, models_kargs["patches"], models_kargs["patches"]),
        proj_drop_rate=models_kargs["dropout"],
        fixed_dropout_depth=models_kargs["fixed_dropout_depth"],
        abs_sincos_enc=models_kargs["abs_sincos_enc"],
        rope_pos_enc=models_kargs["rope_pos_enc"],
    ).to("cuda")

    input_data = get_masked_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs["batch_size"],
        logdir=logdir,
    )


def test_mae_base(models_kargs):

    # clean out existing model
    outdir = models_kargs["outdir"] / "tests/mae/base"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / "logs"
    logdir.mkdir(exist_ok=True, parents=True)

    model = MaskedAutoEncoder(
        model_template="mae-base",
        input_fmt="TZYXC",
        input_shape=inputs[1:],
        patch_shape=(1, 1, models_kargs["patches"], models_kargs["patches"]),
        proj_drop_rate=models_kargs["dropout"],
        abs_sincos_enc=models_kargs["abs_sincos_enc"],
        rope_pos_enc=models_kargs["rope_pos_enc"],
        fixed_dropout_depth=models_kargs["fixed_dropout_depth"],
    ).to("cuda")

    input_data = get_masked_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs["batch_size"],
        logdir=logdir,
    )


def test_mae_large(models_kargs):

    # clean out existing model
    outdir = models_kargs["outdir"] / "tests/mae/large"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / "logs"
    logdir.mkdir(exist_ok=True, parents=True)

    model = MaskedAutoEncoder(
        model_template="mae-large",
        input_fmt="TZYXC",
        input_shape=inputs[1:],
        patch_shape=(1, 1, models_kargs["patches"], models_kargs["patches"]),
        abs_sincos_enc=models_kargs["abs_sincos_enc"],
        rope_pos_enc=models_kargs["rope_pos_enc"],
        proj_drop_rate=models_kargs["dropout"],
        fixed_dropout_depth=models_kargs["fixed_dropout_depth"],
    ).to("cuda")

    input_data = get_masked_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs["batch_size"],
        logdir=logdir,
    )


def test_mae_huge(models_kargs):

    # clean out existing model
    outdir = models_kargs["outdir"] / "tests/mae/huge"
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    inputs = (1, 8, 64, 64, 64, 2)

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)

    logdir = outdir / "logs"
    logdir.mkdir(exist_ok=True, parents=True)

    model = MaskedAutoEncoder(
        model_template="mae-huge",
        input_fmt="TZYXC",
        input_shape=inputs[1:],
        patch_shape=(1, 1, models_kargs["patches"], models_kargs["patches"]),
        proj_drop_rate=models_kargs["dropout"],
        abs_sincos_enc=models_kargs["abs_sincos_enc"],
        rope_pos_enc=models_kargs["rope_pos_enc"],
        fixed_dropout_depth=models_kargs["fixed_dropout_depth"],
    ).to("cuda")

    input_data = get_masked_input_data(model, inputs)

    summarize_model(
        model=model,
        inputs=inputs,
        input_data=input_data,
        batch_size=models_kargs["batch_size"],
        logdir=logdir,
    )
