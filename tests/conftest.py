from pathlib import Path
import pytest

@pytest.fixture(scope="session")
def kargs():
    repo = Path.cwd()
    kargs = dict(
        repo=repo,
        prediction_filename_pattern=r"*[!_gt|!_realspace|!_noisefree|!_predictions_psf|!_corrected_psf|!_reconstructed_psf].tif",
        dataset=repo/"dataset/training_dataset/YuMB_lambda510/z200-y97-x97/z64-y64-x64/z15/",
        outdir=repo/'pretrained_models',
        input_shape=64,
        modes=15,
        batch_size=512,
        hidden_size=768,
        patches=32,
        heads=16,
        repeats=4,
        opt='lamb',
        lr=5e-4,
        wd=5e-5,
        ld=None,
        ema=(.998, 1.),
        epochs=5,
        warmup=1,
        cooldown=1,
        clip_grad=.5,
        fixedlr=False,
        dropout=0.1,
        fixed_dropout_depth=False,
        amp='fp16',
        finetune=None,
        profile=False,
        workers=1,
        gpu_workers=1,
        cpu_workers=8,
    )
    return kargs
