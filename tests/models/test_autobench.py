from types import SimpleNamespace

import pytest
import torch

from cell_observatory_platform.models.meta_arch import autobench
from cell_observatory_platform.utils.registry import REGISTRY


class DummyPatchEmbedding:
    def _unpatchify(self, x, out_channels=None):
        return x + 3


class DummyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = DummyPatchEmbedding()
        self.seen_masks = None

    def forward(self, inputs, masks=None):
        self.seen_masks = masks
        features = inputs + 1
        patches = inputs
        return features, patches


class DummyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_original_patch_indices = None
        self.last_target_masks = None

    def forward(self, x, original_patch_indices=None, target_masks=None):
        self.last_original_patch_indices = original_patch_indices
        self.last_target_masks = target_masks
        return x + 2


def build_dummy_backbone(cfg):
    return DummyBackbone()


def build_dummy_decoder(cfg):
    return DummyDecoder()


# The AutoBench variant __init__ builds its backbone/decoder via
# REGISTRY.build("backbone"/"head", args.name, args), so the dummies are
# registered as name-selected swap points (distinct names from
# test_autobench_build.py to avoid a duplicate-registration clash). Guarded so a
# double import of this module (multiple sys.path roots) doesn't raise.
_DUMMY_BACKBONE = "dummy_autobench_fwd_backbone"
_DUMMY_DECODER = "dummy_autobench_fwd_decoder"
if not REGISTRY.has("backbone", _DUMMY_BACKBONE):
    REGISTRY.register("backbone", _DUMMY_BACKBONE)(build_dummy_backbone)
if not REGISTRY.has("head", _DUMMY_DECODER):
    REGISTRY.register("head", _DUMMY_DECODER)(build_dummy_decoder)

_BACKBONE_ARGS = SimpleNamespace(name=_DUMMY_BACKBONE)

# AutoBench requires output_shape (= train_shape in BUILD): the dense reconstruction's
# channel count follows the DECODER width, not input_shape[-1]. These fixtures use a
# synthetic C=1 input with no mask channel, so the two coincide; DummyPatchEmbedding
# ._unpatchify ignores out_channels anyway, so the value is inert here and exists only
# to satisfy the contract.
_OUTPUT_SHAPE = (1, 1, 1, 1)
_DECODER_ARGS = SimpleNamespace(name=_DUMMY_DECODER)


def test_denoising_forward_uses_aux_loss(monkeypatch):
    recorded = {}

    def dummy_loss_fn(predictions, targets, num_patches, aux_loss_meta=None):
        recorded["predictions"] = predictions
        recorded["targets"] = targets
        recorded["num_patches"] = num_patches
        recorded["aux_loss_meta"] = aux_loss_meta
        return torch.tensor(1.0), {"aux": torch.tensor(2.0)}

    monkeypatch.setattr(autobench, "get_loss_fn", lambda _: dummy_loss_fn)
    monkeypatch.setattr(autobench.AutoBench, "get_num_patches", lambda self: 3)

    model = autobench.DenoisingAutoBench(
        backbone_args=_BACKBONE_ARGS,
        decoder_args=_DECODER_ARGS,
        input_fmt="ZYXC",
        input_shape=(1, 1, 1, 1),
        patch_shape=(1, 1, 1, 1),
        output_shape=_OUTPUT_SHAPE,
        loss_fn="dummy",
        with_auxiliary_loss=True,
    )

    inputs = torch.zeros((2, 3))
    targets = torch.ones_like(inputs)

    loss_dict, predictions = model.forward({"data_tensor": inputs, "metainfo": {"targets": [targets]}})

    assert torch.equal(predictions, inputs + 3), f"Predictions should be inputs + 3, got {predictions} vs expected {inputs + 3}"
    assert torch.equal(loss_dict["step_loss"], torch.tensor(1.0)), f"step_loss should be 1.0, got {loss_dict['step_loss']}"
    assert torch.equal(loss_dict["aux"], torch.tensor(2.0)), f"aux loss should be 2.0, got {loss_dict.get('aux')}"

    assert recorded["num_patches"] == 3, f"num_patches should be 3, got {recorded['num_patches']}"
    assert torch.equal(recorded["targets"], targets), f"Recorded targets should match input targets, got {recorded['targets']} vs {targets}"
    assert torch.equal(recorded["predictions"], predictions), f"Recorded predictions should match output predictions, got {recorded['predictions']} vs {predictions}"
    assert recorded["aux_loss_meta"]["targets"] is targets, "aux_loss_meta['targets'] should be the same object reference as input targets"
    assert recorded["aux_loss_meta"]["predictions"] is predictions, "aux_loss_meta['predictions'] should be the same object reference as output predictions"


def test_denoising_predict_unpatchifies_output(monkeypatch):
    monkeypatch.setattr(autobench, "get_loss_fn", lambda _: lambda *_, **__: (torch.tensor(0.0), None))

    model = autobench.DenoisingAutoBench(
        backbone_args=_BACKBONE_ARGS,
        decoder_args=_DECODER_ARGS,
        input_fmt="ZYXC",
        input_shape=(1, 1, 1, 1),
        patch_shape=(1, 1, 1, 1),
        output_shape=_OUTPUT_SHAPE,
        loss_fn="dummy",
    )

    inputs = torch.zeros((1, 2))
    # inference_step returns the dense prediction keyed by task name; evaluate_step
    # would return patch-space (+3), not the unpatchified +6 this test is named for.
    output = model.inference_step({"data_tensor": inputs, "metainfo": {}})[model.task]

    expected = inputs + 6
    assert torch.equal(output, expected), f"Predict output should be inputs + 6, got {output} vs expected {expected}"


def test_channel_split_requires_channel_last(monkeypatch):
    monkeypatch.setattr(autobench, "get_loss_fn", lambda _: lambda *_, **__: (torch.tensor(0.0), None))

    with pytest.raises(ValueError):
        autobench.ChannelSplitAutoBench(
            backbone_args=_BACKBONE_ARGS,
            decoder_args=_DECODER_ARGS,
            input_fmt="TZYX",
            input_shape=(1, 1, 1, 1),
            patch_shape=(1, 1, 1, 1),
            output_shape=_OUTPUT_SHAPE,
            loss_fn="dummy",
        )


def test_upsample_time_forward_masks_targets(monkeypatch):
    def dummy_apply_masks(tensor, masks=None):
        return tensor + 5

    def dummy_loss_fn(predictions, targets, num_patches, aux_loss_meta=None):
        return torch.tensor(7.0), None

    monkeypatch.setattr(autobench, "apply_masks", dummy_apply_masks)
    monkeypatch.setattr(autobench, "get_loss_fn", lambda _: dummy_loss_fn)
    monkeypatch.setattr(autobench.AutoBench, "get_num_patches", lambda self: 11)

    model = autobench.UpsampleTimeAutoBench(
        backbone_args=_BACKBONE_ARGS,
        decoder_args=_DECODER_ARGS,
        input_fmt="ZYXC",
        input_shape=(1, 1, 1, 1),
        patch_shape=(1, 1, 1, 1),
        output_shape=_OUTPUT_SHAPE,
        loss_fn="dummy",
    )

    inputs = torch.zeros((1, 2))
    context_masks = torch.ones((1, 1), dtype=torch.bool)
    target_masks = torch.ones_like(context_masks)
    original_patch_indices = torch.arange(1)

    loss_dict, predictions = model.forward(
        {
            "data_tensor": inputs,
            "metainfo": {
                "context_masks": [context_masks],
                "target_masks": [target_masks],
                "original_patch_indices": [original_patch_indices],
            },
        }
    )

    assert isinstance(predictions, torch.Tensor), f"Predictions should be a Tensor, got {type(predictions)}"
    assert type(model.decoder).__name__ == "DummyDecoder", f"Decoder should be DummyDecoder, got {type(model.decoder).__name__}"
    assert hasattr(model.decoder, "last_original_patch_indices"), "Decoder should have last_original_patch_indices attribute"
    assert hasattr(model.decoder, "last_target_masks"), "Decoder should have last_target_masks attribute"
    decoder: DummyDecoder = model.decoder  # type: ignore[assignment]
    assert torch.equal(predictions, inputs + 8), f"Predictions should be inputs + 8, got {predictions} vs expected {inputs + 8}"
    assert torch.is_tensor(loss_dict["step_loss"]), f"step_loss should be a tensor, got {type(loss_dict['step_loss'])}"
    assert loss_dict["step_loss"].item() == pytest.approx(7.0), f"step_loss should be approximately 7.0, got {loss_dict['step_loss'].item()}"
    assert decoder.last_original_patch_indices is not None, "decoder.last_original_patch_indices should not be None"
    assert decoder.last_target_masks is not None, "decoder.last_target_masks should not be None"
    assert torch.equal(decoder.last_original_patch_indices, original_patch_indices), f"decoder.last_original_patch_indices should match input, got {decoder.last_original_patch_indices} vs {original_patch_indices}"
    assert torch.equal(decoder.last_target_masks, target_masks), f"decoder.last_target_masks should match input, got {decoder.last_target_masks} vs {target_masks}"

