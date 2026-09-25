import pytest

from cell_observatory_platform.models.backbones.dino_encoder import DinoEncoder


def test_dino_encoder_rejects_non_custom_rope_type():
    """DinoEncoder always carries prefix tokens (cls + registers); only the
    'custom' rope slices them, so axial/mixed rope is refused at construction."""
    with pytest.raises(ValueError, match="rope_type='custom'"):
        DinoEncoder(
            input_format="ZYXC",
            input_shape=(8, 8, 8, 2),
            patch_shape=(4, 4, 4),
            embed_dim=24,
            depth=1,
            num_heads=2,
            rope_type="axial",
        )
