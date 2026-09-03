from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from cell_observatory_platform.models.meta_arch import autobench
from cell_observatory_platform.training.losses import L2_masked_loss
from cell_observatory_platform.utils.registry import REGISTRY


class DummyPatchEmbedding:
    def __init__(self):
        self.seen_out_channels = "never-called"

    def _unpatchify(self, x, out_channels=None):
        self.seen_out_channels = out_channels
        return x + 3


class DummyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = DummyPatchEmbedding()
        self.seen_masks = None
        self.scale = torch.nn.Parameter(torch.ones(1))  # so freeze_backbone has something to freeze

    def forward(self, inputs, masks=None):
        self.seen_masks = masks
        return inputs + 1, inputs


class DummyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_original_patch_indices = None
        self.last_target_masks = None
        self.bias = torch.nn.Parameter(torch.zeros(1))

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
# ._unpatchify records out_channels so the inference tests can check what was passed.
_OUTPUT_SHAPE = (1, 1, 1, 1)
_DECODER_ARGS = SimpleNamespace(name=_DUMMY_DECODER)

_VARIANTS = [
    autobench.DenoisingAutoBench,
    autobench.ChannelSplitAutoBench,
    autobench.UpsampleTimeAutoBench,
    autobench.UpsampleSpaceAutoBench,
    autobench.UpsampleSpaceTimeAutoBench,
]


def test_denoising_forward_uses_aux_loss(monkeypatch):
    recorded = {}

    def dummy_loss_fn(predictions, targets, num_patches, aux_loss_meta=None):
        recorded["predictions"] = predictions
        recorded["targets"] = targets
        recorded["num_patches"] = num_patches
        recorded["aux_loss_meta"] = aux_loss_meta
        return torch.tensor(1.0), {"aux": torch.tensor(2.0)}

    monkeypatch.setattr(autobench, "get_loss_fn", lambda _: dummy_loss_fn)

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

    loss_dict, predictions = model.forward(
        {"data_tensor": inputs, "metainfo": {"targets": {"denoising": targets}}}   # Form D
    )

    assert torch.equal(predictions, inputs + 3), f"Predictions should be inputs + 3, got {predictions} vs expected {inputs + 3}"
    assert torch.equal(loss_dict["step_loss"], torch.tensor(1.0)), f"step_loss should be 1.0, got {loss_dict['step_loss']}"
    assert torch.equal(loss_dict["aux"], torch.tensor(2.0)), f"aux loss should be 2.0, got {loss_dict.get('aux')}"

    # num_patches is now the batch-wide supervised count B * N (= 2 * 3 here),
    # NOT the per-sample get_num_patches() — the loss sums over the batch.
    assert recorded["num_patches"] == targets.shape[0] * targets.shape[1], (
        f"num_patches should be B*N={targets.shape[0] * targets.shape[1]}, got {recorded['num_patches']}"
    )
    assert torch.equal(recorded["targets"], targets), f"Recorded targets should match input targets, got {recorded['targets']} vs {targets}"
    assert torch.equal(recorded["predictions"], predictions), f"Recorded predictions should match output predictions, got {recorded['predictions']} vs {predictions}"
    assert recorded["aux_loss_meta"]["targets"] is targets, "aux_loss_meta['targets'] should be the same object reference as input targets"
    assert recorded["aux_loss_meta"]["predictions"] is predictions, "aux_loss_meta['predictions'] should be the same object reference as output predictions"


@pytest.mark.parametrize("variant_cls", _VARIANTS, ids=lambda c: c.__name__)
def test_inference_step_unpatchifies_with_output_channels(variant_cls, monkeypatch):
    """The dense reconstruction is unpatchified with output_shape[-1] (the decoder's
    channel count), not the input C, and is returned under the task key only."""
    monkeypatch.setattr(autobench, "get_loss_fn", lambda _: lambda *_, **__: (torch.tensor(0.0), None))
    model = variant_cls(
        backbone_args=_BACKBONE_ARGS,
        decoder_args=_DECODER_ARGS,
        input_fmt="ZYXC",
        input_shape=(1, 1, 1, 2),   # input C=2
        patch_shape=(1, 1, 1, 1),
        output_shape=(1, 1, 1, 1),  # output C=1 (mask channel stripped)
        loss_fn="dummy",
    )
    inputs = torch.zeros(1, 2)
    out = model.inference_step({"data_tensor": inputs, "metainfo": {}})
    assert model.backbone.patch_embedding.seen_out_channels == 1
    assert set(out) == {model.task}
    assert torch.equal(out[model.task], inputs + 6)  # backbone +1, decoder +2, unpatchify +3


@pytest.mark.parametrize("variant_cls", _VARIANTS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("freeze", [True, False])
def test_freeze_backbone_freezes_backbone_params_in_every_variant(variant_cls, freeze, monkeypatch):
    """freeze_backbone=True leaves every backbone parameter with requires_grad False
    in every variant; False leaves them trainable. The decoder is never frozen."""
    monkeypatch.setattr(autobench, "get_loss_fn", lambda _: lambda *_, **__: (torch.tensor(0.0), None))
    model = variant_cls(
        backbone_args=_BACKBONE_ARGS,
        decoder_args=_DECODER_ARGS,
        input_fmt="ZYXC",
        input_shape=(1, 1, 1, 1),
        patch_shape=(1, 1, 1, 1),
        output_shape=_OUTPUT_SHAPE,
        loss_fn="dummy",
        freeze_backbone=freeze,
    )
    backbone_params = list(model.backbone.parameters())
    decoder_params = list(model.decoder.parameters())
    assert backbone_params and decoder_params
    assert all(p.requires_grad is (not freeze) for p in backbone_params)
    assert all(p.requires_grad for p in decoder_params)


def test_finalize_build_freezes_backbone_params():
    """_finalize_build applies the freeze_backbone flag to whatever backbone is attached."""
    b = object.__new__(autobench.DenoisingAutoBench)
    nn.Module.__init__(b)
    b.freeze_backbone = True
    b.backbone = nn.Linear(4, 4)
    b._finalize_build()
    assert all(not p.requires_grad for p in b.backbone.parameters())


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


def test_upsample_time_forward_supervises_only_target_patches(monkeypatch):
    """Loss sees the decoder output and the raw patches gathered at target_masks only."""
    recorded = {}

    def dummy_loss_fn(predictions, targets, num_patches, aux_loss_meta=None):
        recorded.update(predictions=predictions, targets=targets, num_patches=num_patches)
        return torch.tensor(7.0), None

    monkeypatch.setattr(autobench, "get_loss_fn", lambda _: dummy_loss_fn)
    model = autobench.UpsampleTimeAutoBench(
        backbone_args=_BACKBONE_ARGS,
        decoder_args=_DECODER_ARGS,
        input_fmt="ZYXC",
        input_shape=(1, 1, 1, 1),
        patch_shape=(1, 1, 1, 1),
        output_shape=_OUTPUT_SHAPE,
        loss_fn="dummy",
    )
    inputs = torch.arange(8.0).view(1, 4, 2)  # (B, N=4 patches, D)
    context_masks = torch.tensor([[1, 3]])
    target_masks = torch.tensor([[0, 2]])
    original_patch_indices = torch.arange(4)

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

    # backbone (+1) then decoder (+2) on the full sequence; only target patches are kept
    assert torch.equal(predictions, inputs[:, [0, 2]] + 3)
    assert torch.equal(recorded["predictions"], predictions)
    assert torch.equal(recorded["targets"], inputs[:, [0, 2]])  # targets = raw patches at target_masks
    assert recorded["num_patches"] == 1 * 2
    assert loss_dict["step_loss"].item() == pytest.approx(7.0)
    assert torch.equal(model.backbone.seen_masks, context_masks)
    assert torch.equal(model.decoder.last_target_masks, target_masks)
    assert torch.equal(model.decoder.last_original_patch_indices, original_patch_indices)


def _stub_autobench(cls):
    m = cls.__new__(cls)
    nn.Module.__init__(m)
    m.with_auxiliary_loss = False
    m.target_role = "recon"        # Form-D role the fake sample publishes under
    m.loss_fn = L2_masked_loss
    # patches (2nd return) = inputs + 1 so the patch-supervised task
    # (UpsampleTime, whose targets come from `patches`) sees unit error too
    m.backbone = lambda inputs, **kw: (inputs, inputs + 1.0)
    m.decoder = lambda x, **kw: x
    return m


def _sample(B, N=10, D=4, err=1.0):
    x = torch.randn(1, N, D).repeat(B, 1, 1)
    targets = x + err
    return {
        "data_tensor": x,
        "metainfo": {
            "targets": {"recon": targets},   # Form D (see data/data_types.py)
            "target_masks": [torch.arange(N // 2).unsqueeze(0).expand(B, -1)],
        },
    }


@pytest.mark.parametrize("variant_cls", _VARIANTS, ids=lambda c: c.__name__)
def test_loss_is_invariant_to_batch_size(variant_cls):
    """A unit per-element error gives step_loss == 1 regardless of batch size:
    the loss is normalised by the batch-wide supervised patch count."""
    losses = []
    for B in (1, 2, 4):
        m = _stub_autobench(variant_cls)
        loss_dict, _ = m.forward(_sample(B))
        losses.append(float(loss_dict["step_loss"]))
    assert losses[0] == pytest.approx(1.0, rel=1e-4), losses
    assert losses[0] == pytest.approx(losses[1], rel=1e-6)
    assert losses[0] == pytest.approx(losses[2], rel=1e-6)
