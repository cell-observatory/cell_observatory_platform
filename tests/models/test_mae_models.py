import pytest
import torch

from cell_observatory_platform.models.meta_arch.maskedautoencoder import MaskedAutoEncoder
from cell_observatory_platform.training.helpers import get_masked_input_data


def test_mae_forward_loss_and_prediction_shapes():
    """The decoder reconstructs raw pixels for the masked-out patches only; the loss
    is a finite positive scalar that back-propagates into the decoder."""
    torch.manual_seed(0)
    inputs = (1, 16, 16, 16, 1)
    model = MaskedAutoEncoder(
        model_template="mae",
        input_fmt="ZYXC",
        input_shape=inputs[1:],
        patch_shape=(4, 4, 4, None),
        embed_dim=16,
        decoder_embed_dim=16,
        depth=1,
        decoder_depth=1,
        num_heads=2,
        decoder_num_heads=2,
        drop_path_rate=0.0,
        abs_sincos_enc=True,
        rope_pos_enc=False,
        dtype=torch.float32,
        buffer_device="cpu",
    )
    (data_sample,) = get_masked_input_data(model, inputs, device="cpu", mask_ratio=0.75)
    n_patches = model.get_num_patches()
    n_target = n_patches - int(n_patches * 0.25)
    pixels_per_patch = model.masked_encoder.patch_embedding.pixels_per_patch  # 4*4*4*1

    loss_dict, predictions = model(data_sample)

    assert predictions.shape == (1, n_target, pixels_per_patch)
    assert torch.isfinite(predictions).all()
    loss = loss_dict["step_loss"]
    assert loss.ndim == 0 and torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    grads = [p.grad for p in model.masked_decoder.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)


def test_masked_autoencoder_rejects_use_deformable_attn():
    """Deformable attention on the masked encoder is unsupported (it needs the
    mask_indices scatter metadata) and must be refused at construction."""
    with pytest.raises(NotImplementedError, match="mask_indices"):
        MaskedAutoEncoder(
            model_template="mae",
            input_fmt="ZYXC",
            input_shape=(32, 64, 64, 1),
            patch_shape=(4, 8, 8),
            embed_dim=32,
            depth=1,
            num_heads=2,
            decoder_embed_dim=32,
            decoder_depth=1,
            decoder_num_heads=2,
            use_deformable_attn=True,
        )
