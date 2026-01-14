import copy

import pytest
import torch
from omegaconf import OmegaConf

from cell_observatory_platform.models.meta_arch import autobench


class DummyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()


class DummyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()


def build_dummy_backbone(cfg):
    return DummyBackbone()


def build_dummy_decoder(cfg):
    return DummyDecoder()


def _make_cfg(*, task: str, input_fmt: str = "ZYXC"):
    model_cfg = {
        "backbone_args": {
            "BUILD": "tests.models.test_autobench_build.build_dummy_backbone",
            # Intentionally on backbone_args to exercise the fallback path:
            # embed_dim = model_cfg.get("embed_dim", backbone_args.get("embed_dim", None))
            "embed_dim": 8,
        },
        "decoder_args": {
            "BUILD": "tests.models.test_autobench_build.build_dummy_decoder",
        },
        "input_fmt": input_fmt,
        "input_shape": [4, 4, 4, 2],  # Z, Y, X, C (only used when input_fmt == "ZYXC")
        "patch_shape": [2, 2],  # axial, lateral
        "loss_fn": "l2_masked",
        "abs_sincos_enc": False,
        "weight_init_type": "mae",
        "with_auxiliary_loss": False,
    }

    cfg = OmegaConf.create(
        {
            "tasks": {"task": task},
            "models": {
                "meta_arch": {
                    "autobench": {
                        "DenoisingAutoBench": copy.deepcopy(model_cfg),
                        "ChannelSplitAutoBench": copy.deepcopy(model_cfg),
                        "UpsampleTimeAutoBench": copy.deepcopy(model_cfg),
                        "UpsampleSpaceAutoBench": copy.deepcopy(model_cfg),
                        "UpsampleSpaceTimeAutoBench": copy.deepcopy(model_cfg),
                    }
                }
            },
        }
    )

    # Exercise the struct toggle path in BUILD() (decoder_args gets mutated).
    task_to_key = {
        "denoising": "DenoisingAutoBench",
        "channel_split": "ChannelSplitAutoBench",
        "upsample_time": "UpsampleTimeAutoBench",
        "upsample_space": "UpsampleSpaceAutoBench",
        "upsample_spacetime": "UpsampleSpaceTimeAutoBench",
    }
    decoder_args = cfg.models.meta_arch.autobench[task_to_key[task]].decoder_args
    OmegaConf.set_struct(decoder_args, True)

    return cfg


@pytest.mark.parametrize(
    "task, expected_cls",
    [
        ("denoising", autobench.DenoisingAutoBench),
        ("channel_split", autobench.ChannelSplitAutoBench),
        ("upsample_time", autobench.UpsampleTimeAutoBench),
        ("upsample_space", autobench.UpsampleSpaceAutoBench),
        ("upsample_spacetime", autobench.UpsampleSpaceTimeAutoBench),
    ],
)
def test_BUILD_dispatch_and_injects_decoder_dims(task, expected_cls):
    cfg = _make_cfg(task=task)
    model = autobench.BUILD(cfg)

    assert isinstance(model, expected_cls), (
        f"BUILD() should return {expected_cls.__name__} for task={task}, got {type(model).__name__}"
    )

    assert model.decoder_args["input_dim"] == 8, (
        f"decoder_args.input_dim should be derived from backbone_args.embed_dim=8, got {model.decoder_args.get('input_dim')}"
    )

    expected_output_dim = 2 * 2 * (2**2)  # C * axial * lateral^2
    assert model.decoder_args["output_dim"] == expected_output_dim, (
        f"decoder_args.output_dim should be C * axial * lateral^2 = {expected_output_dim}, got {model.decoder_args.get('output_dim')}"
    )

    assert OmegaConf.is_struct(model.decoder_args), "decoder_args should remain struct=True after BUILD()"


def test_BUILD_rejects_non_ZYXC_input_fmt():
    cfg = _make_cfg(task="denoising", input_fmt="TZYXC")
    with pytest.raises(ValueError, match="only supports 'ZYXC'"):
        autobench.BUILD(cfg)


