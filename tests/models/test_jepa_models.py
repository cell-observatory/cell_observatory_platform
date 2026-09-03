import torch

from cell_observatory_platform.models.meta_arch.jepa import JEPA
from cell_observatory_platform.training.helpers import get_masked_input_data


def test_jepa_forward_loss_and_prediction_shapes():
    """The predictor output is gathered at the target patches and projected back to
    embed_dim; the loss is a finite positive scalar that back-propagates into the
    predictor, while the EMA target encoder is never trainable."""
    torch.manual_seed(0)
    inputs = (1, 16, 16, 16, 1)  # (B, Z, Y, X, C); get_masked_input_data builds B=1 masks
    model = JEPA(
        input_fmt="ZYXC",
        input_shape=inputs[1:],
        patch_shape=(4, 4, 4, None),
        embed_dim=16,
        predictor_embed_dim=16,
        depth=1,
        predictor_depth=1,
        num_heads=2,
        predictor_num_heads=2,
        drop_path_rate=0.0,
        abs_sincos_enc=True,
        rope_pos_enc=False,
        dtype=torch.float32,
        buffer_device="cpu",
    )
    (data_sample,) = get_masked_input_data(model, inputs, device="cpu", mask_ratio=0.75)
    n_patches = model.get_num_patches()            # 64
    n_target = n_patches - int(n_patches * 0.25)   # patches the predictor must reconstruct

    loss_dict, predictions = model(data_sample)

    assert predictions.shape == (1, n_target, model.embed_dim)
    assert torch.isfinite(predictions).all()
    loss = loss_dict["step_loss"]
    assert loss.ndim == 0 and torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    grads = [p.grad for p in model.target_predictor.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    # target encoder is an EMA copy, never trained
    assert all(not p.requires_grad for p in model.target_encoder.parameters())
