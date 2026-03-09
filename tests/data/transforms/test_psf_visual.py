import os
from pathlib import Path

import pytest
import torch

from cell_observatory_platform.data.transforms.psf import ConvolveWithPSF


def _make_cell_like_volume(
    shape_zyx: tuple[int, int, int],
    *,
    cell_radius: float = 10.0,
    nucleus_radius: float = 4.0,
    cell_intensity: float = 1.0,
    nucleus_intensity: float = 2.0,
) -> torch.Tensor:
    """Simple synthetic 'cell' with a brighter nucleus: (Z,Y,X) float32 volume."""
    z, y, x = shape_zyx
    zz, yy, xx = torch.meshgrid(
        torch.arange(z, dtype=torch.float32),
        torch.arange(y, dtype=torch.float32),
        torch.arange(x, dtype=torch.float32),
        indexing="ij",
    )
    cz, cy, cx = (z - 1) / 2.0, (y - 1) / 2.0, (x - 1) / 2.0
    rr = torch.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)

    cell = (rr <= cell_radius).to(torch.float32) * cell_intensity
    nucleus = (rr <= nucleus_radius).to(torch.float32) * nucleus_intensity
    return cell + nucleus


def _tv3(x: torch.Tensor) -> torch.Tensor:
    """3D total variation proxy (sum of absolute finite differences)."""
    dz = (x[1:, :, :] - x[:-1, :, :]).abs().sum()
    dy = (x[:, 1:, :] - x[:, :-1, :]).abs().sum()
    dx = (x[:, :, 1:] - x[:, :, :-1]).abs().sum()
    return dz + dy + dx


def _save_orthoslices(volume_zyx: torch.Tensor, outpath: Path, *, title: str) -> None:
    """
    Save orthoslices (XY, XZ, YZ) of a 3D volume.

    Notes:
    - Uses percentile normalization for readability.
    - Intended for optional/visual debugging (not for strict numerical assertions).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    arr = volume_zyx.detach().cpu().numpy()
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume (ZYX); got shape {arr.shape}")

    zc, yc, xc = [s // 2 for s in arr.shape]
    arr_flat = arr.flatten()
    vmin = 0
    vmax = arr.max()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title)

    im0 = ax[0].imshow(arr[zc, :, :], cmap="gray", vmin=vmin, vmax=vmax)
    ax[0].set_title(f"XY z={zc}")
    plt.colorbar(im0, fraction=0.046, pad=0.04)
    im1 = ax[1].imshow(arr[:, yc, :], cmap="gray", vmin=vmin, vmax=vmax)
    ax[1].set_title(f"XZ y={yc}")
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    im2 = ax[2].imshow(arr[:, :, xc], cmap="gray", vmin=vmin, vmax=vmax)
    ax[2].set_title(f"YZ x={xc}")
    plt.colorbar(im2, fraction=0.046, pad=0.04)

    for a in ax:
        a.axis("off")

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)


@pytest.mark.skip(reason="Just for local inspection")
def test_cell_like_object_gets_smoother_after_psf_convolution():
    """
    Optional/qualitative test:
    - Convolution with a normalized PSF should preserve total intensity (sum).
    - It should reduce high-frequency content; we use total-variation as a robust proxy.

    This isn't meant to validate exact optics, just catch obvious regressions in padding/FFT/normalization.
    """
    # If set, write plots here (useful for quick local inspection).
    # Example:
    #   CELL_OBS_PSF_VIZ_DIR=/tmp/psf_viz pytest -m slow -s tests/data/transforms/test_psf_celllike.py
    viz_dir = Path("/clusterfs/nvme/martinalvarez/GitHub/scratch") / "psf_celllike_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    psf = torch.exp(
        -(
            (torch.arange(9)[:, None, None] - 4) ** 2
            + (torch.arange(9)[None, :, None] - 4) ** 2
            + (torch.arange(9)[None, None, :] - 4) ** 2
        )
        / (2 * 1.5**2)
    ).to(torch.float32)
    psf = psf / psf.sum()

    vol_zyx = _make_cell_like_volume((33, 33, 33), cell_radius=10.0, nucleus_radius=4.0)
    data = vol_zyx[None, ..., None].contiguous()  # BZYXC

    transform = ConvolveWithPSF(
        psf=psf,
        pad_type="reflect",
        input_format="ZYXC",
        input_shape=tuple(data.shape[1:]),
        input_pixel_size_um=(1.0, 1.0, 1.0),
        psf_format="ZYX",
        psf_pixel_size_um=(1.0, 1.0, 1.0),
        psf_centered=True,
        visualization_dir=str(viz_dir),
    )

    out = transform(data)[0, :, :, :, 0]

    # Save input/output orthoslice views for visual inspection.
    _save_orthoslices(vol_zyx, viz_dir / "cell_input.png", title="Cell-like input")
    _save_orthoslices(out, viz_dir / "cell_output.png", title="After PSF convolution")
    _save_orthoslices((out - vol_zyx).abs(), viz_dir / "cell_absdiff.png", title="|Output - Input|")

    # With a normalized PSF and reflect padding, total intensity should be preserved.
    assert torch.isclose(out.sum(), vol_zyx.sum(), rtol=1e-4, atol=1e-4)

    # Convolution should reduce sharp edges -> TV should decrease.
    assert _tv3(out) < _tv3(vol_zyx)


