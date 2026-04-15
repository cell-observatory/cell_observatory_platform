"""
Unit tests for zarr IO functions in cell_observatory_platform.data.io.

Tests cover the full zarr IO lifecycle:
  - Creating zarr stores (save_zarr_data)
  - Reading zarr data back (read_zarr)
  - Appending mask channels to root array (update_zarr_data mode="append")
  - Overwriting mask channels selectively (update_zarr_data mode="overwrite")
  - Creating label arrays under <source>/<label> groups (save_zarr_labels)
  - Creating or overwriting label arrays via save_zarr_annotations (overwrite upserts)
  - High-level save_masks orchestration
  - Existence checks (annotation_exists)
  - Shape normalization helpers (normalize_data_shape, normalize_idxs)

All spatial tests are parameterized for both ZYXC (3D+C) and TZYXC (4D+C) layouts.
"""
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest
import tensorstore as ts

from cell_observatory_platform.data.io import (
    _make_read_zarr_spec,
    _make_write_zarr_spec,
    annotation_exists,
    create_zarr_spec,
    normalize_idxs,
    read_channel_names,
    read_zarr,
    save_masks,
    save_zarr_data,
    save_zarr_annotations,
    update_zarr_data,
)

# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------

SHARD_SPATIAL_SHAPE = (4, 4, 4)
CHUNK_SPATIAL_SHAPE = (2, 2, 2)
SPATIAL = (8, 16, 16)  # Z, Y, X
ZARR_DRIVER = "zarr3"
DTYPE = "uint16"


def _make_data(data_format: str, n_channels: int = 2, n_timepoints: int = 3, spatial: Tuple = SPATIAL) -> np.ndarray:
    """Build a synthetic data array matching the given format."""
    z, y, x = spatial
    if data_format == "TZYXC":
        shape = (n_timepoints, z, y, x, n_channels)
    elif data_format == "ZYXC":
        shape = (z, y, x, n_channels)
    else:
        raise ValueError(f"Unsupported format: {data_format}")
    rng = np.random.default_rng(42)
    return rng.uniform(1.0, 100.0, size=shape).astype(np.float32)


def _make_mask(data_format: str, n_channels: int = 1, n_timepoints: int = 3, spatial: Tuple = SPATIAL) -> np.ndarray:
    z, y, x = spatial
    if data_format == "TZYXC":
        shape = (n_timepoints, z, y, x, n_channels)
    elif data_format == "ZYXC":
        shape = (z, y, x, n_channels)
    else:
        raise ValueError(f"Unsupported format: {data_format}")
    rng = np.random.default_rng(99)
    return rng.integers(0, 5, size=shape).astype(np.uint16)


def _read_root(path: str) -> np.ndarray:
    ds = ts.open(_make_read_zarr_spec(path, driver=ZARR_DRIVER), read=True).result()
    return ds.read().result()


def _read_annotation(path: str, source: str, annotation: str) -> np.ndarray:
    ds = ts.open(
        _make_read_zarr_spec(path, subpath=f"{source}/{annotation}", driver=ZARR_DRIVER),
        read=True,
    ).result()
    return ds.read().result()


def _to_disk(arr: np.ndarray, data_format: str) -> np.ndarray:
    """Expand a non-T array to its expected on-disk (T-bearing) shape for comparison."""
    if "T" in data_format:
        return arr
    return arr[np.newaxis, ...]


def _tp(data_format: str) -> Optional[list]:
    """Return ``timepoint_idxs`` required for non-T formats, else ``None``."""
    if "T" in data_format:
        return None
    return [0]


# ---------------------------------------------------------------------------
# Parametrize across 3D (ZYXC) and 4D (TZYXC) formats
# ---------------------------------------------------------------------------

FORMAT_PARAMS = pytest.mark.parametrize("data_format", ["ZYXC", "TZYXC"], ids=["3D-ZYXC", "4D-TZYXC"])


# ===========================================================================
# 1. Spec construction
# ===========================================================================

class TestSpecConstruction:
    def test_read_spec_minimal(self):
        spec = _make_read_zarr_spec("/some/path.zarr", driver="zarr3")
        assert spec["driver"] == "zarr3"
        assert spec["kvstore"]["path"] == "/some/path.zarr"
        assert spec["path"] == ""

    def test_read_spec_with_subpath(self):
        spec = _make_read_zarr_spec("/p.zarr", subpath="model/labels", driver="zarr3")
        assert spec["path"] == "model/labels"

    def test_write_spec_zarr3_has_sharding(self):
        spec = _make_write_zarr_spec(
            data_shape=(1, 8, 16, 16, 2),
            zarr_version="zarr3",
            path="/p.zarr",
            shard_shape=(1, 4, 4, 4, 2),
            chunk_shape=(1, 2, 2, 2, 2),
            dtype="uint16",
        )
        assert spec["driver"] == "zarr3"
        codecs = spec["metadata"]["codecs"]
        assert any(c["name"] == "sharding_indexed" for c in codecs)
        assert spec["metadata"]["fill_value"] == 0

    @FORMAT_PARAMS
    def test_create_zarr_spec_roundtrip(self, data_format):
        data = _make_data(data_format)
        disk_data = _to_disk(data, data_format)
        disk_format = f"T{data_format}" if "T" not in data_format else data_format
        spec = create_zarr_spec(
            data_shape=disk_data.shape,
            zarr_version=ZARR_DRIVER,
            path="/p.zarr",
            data_format=disk_format,
            shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE,
            dtype=DTYPE,
        )
        assert tuple(spec["metadata"]["shape"]) == tuple(disk_data.shape)
        assert len(spec["metadata"]["chunk_grid"]["configuration"]["chunk_shape"]) == len(disk_data.shape)


# ===========================================================================
# 2. save_zarr_data & read_zarr round-trip
# ===========================================================================

class TestSaveAndReadZarrData:
    @FORMAT_PARAMS
    def test_create_and_read_back(self, tmp_path, data_format):
        data = _make_data(data_format)
        zarr_path = str(tmp_path / "test.zarr")

        save_zarr_data(
            image_path=zarr_path,
            data=data,
            shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE,
            data_format=data_format,
            zarr_driver=ZARR_DRIVER,
            dtype=DTYPE,
        )

        stored = _read_root(zarr_path)
        np.testing.assert_array_equal(stored, _to_disk(data, data_format).astype(np.uint16))

    @FORMAT_PARAMS
    def test_create_refuses_existing_path(self, tmp_path, data_format):
        data = _make_data(data_format)
        zarr_path = str(tmp_path / "exists.zarr")

        save_zarr_data(
            image_path=zarr_path,
            data=data,
            shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE,
            data_format=data_format,
            zarr_driver=ZARR_DRIVER,
            dtype=DTYPE,
        )

        with pytest.raises(FileExistsError):
            save_zarr_data(
                image_path=zarr_path,
                data=data,
                shard_spatial_shape=SHARD_SPATIAL_SHAPE,
                chunk_spatial_shape=CHUNK_SPATIAL_SHAPE,
                data_format=data_format,
                zarr_driver=ZARR_DRIVER,
                dtype=DTYPE,
            )
    


# ===========================================================================
# 2b. save_zarr_data — time_dim_size / timepoint_idxs validation
# ===========================================================================

class TestSaveZarrDataTimeArgs:
    def test_zyxc_only_time_dim_size_raises(self, tmp_path):
        data = _make_data("ZYXC")
        with pytest.raises(ValueError, match="both be provided or both omitted"):
            save_zarr_data(
                image_path=str(tmp_path / "a.zarr"), data=data,
                shard_spatial_shape=SHARD_SPATIAL_SHAPE,
                chunk_spatial_shape=CHUNK_SPATIAL_SHAPE,
                data_format="ZYXC", time_dim_size=5,
            )

    def test_zyxc_only_timepoint_idxs_raises(self, tmp_path):
        data = _make_data("ZYXC")
        with pytest.raises(ValueError, match="both be provided or both omitted"):
            save_zarr_data(
                image_path=str(tmp_path / "a.zarr"), data=data,
                shard_spatial_shape=SHARD_SPATIAL_SHAPE,
                chunk_spatial_shape=CHUNK_SPATIAL_SHAPE,
                data_format="ZYXC", timepoint_idxs=[2],
            )

    def test_zyxc_both_provided_ok(self, tmp_path):
        data = _make_data("ZYXC")
        save_zarr_data(
            image_path=str(tmp_path / "a.zarr"), data=data,
            shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE,
            data_format="ZYXC", time_dim_size=5, timepoint_idxs=[2],
        )
        stored = _read_root(str(tmp_path / "a.zarr"))
        assert stored.shape[0] == 5

    def test_zyxc_neither_provided_defaults_unitary(self, tmp_path):
        data = _make_data("ZYXC")
        save_zarr_data(
            image_path=str(tmp_path / "a.zarr"), data=data,
            shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE,
            data_format="ZYXC",
        )
        stored = _read_root(str(tmp_path / "a.zarr"))
        assert stored.shape[0] == 1

    def test_tzyxc_time_dim_size_alone_ok(self, tmp_path):
        """T-bearing formats should not enforce the both-or-neither rule."""
        n_t = 3
        data = _make_data("TZYXC", n_timepoints=n_t)
        save_zarr_data(
            image_path=str(tmp_path / "a.zarr"), data=data,
            shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE,
            data_format="TZYXC", time_dim_size=n_t,
        )


# ===========================================================================
# 3. update_zarr_data — append mode
# ===========================================================================

class TestUpdateZarrAppend:
    def _create_store(self, zarr_path: str, data: np.ndarray, data_format: str):
        save_zarr_data(
            image_path=zarr_path,
            data=data,
            shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE,
            data_format=data_format,
            zarr_driver=ZARR_DRIVER,
            dtype=DTYPE,
        )

    @FORMAT_PARAMS
    def test_append_single_mask_channel(self, tmp_path, data_format):
        data = _make_data(data_format, n_channels=2)
        zarr_path = str(tmp_path / "img.zarr")
        self._create_store(zarr_path, data, data_format)

        mask = _make_mask(data_format, n_channels=1)
        update_zarr_data(
            image_path=zarr_path,
            data=mask,
            data_format=data_format,
            zarr_driver=ZARR_DRIVER,
            dtype=DTYPE,
            timepoint_idxs=_tp(data_format),
            mode="append",
        )

        stored = _read_root(zarr_path)
        assert stored.shape[-1] == 3  # 2 data + 1 mask
        np.testing.assert_array_equal(stored[..., :2], _to_disk(data, data_format).astype(np.uint16))
        np.testing.assert_array_equal(stored[..., 2:], _to_disk(mask, data_format).astype(np.uint16))

    @FORMAT_PARAMS
    def test_append_multiple_mask_channels(self, tmp_path, data_format):
        data = _make_data(data_format, n_channels=2)
        zarr_path = str(tmp_path / "img.zarr")
        self._create_store(zarr_path, data, data_format)

        mask = _make_mask(data_format, n_channels=3)
        update_zarr_data(
            image_path=zarr_path,
            data=mask,
            data_format=data_format,
            zarr_driver=ZARR_DRIVER,
            dtype=DTYPE,
            timepoint_idxs=_tp(data_format),
            mode="append",
        )

        stored = _read_root(zarr_path)
        assert stored.shape[-1] == 5
        np.testing.assert_array_equal(stored[..., :2], _to_disk(data, data_format).astype(np.uint16))
        np.testing.assert_array_equal(stored[..., 2:], _to_disk(mask, data_format).astype(np.uint16))

    @FORMAT_PARAMS
    def test_append_with_timepoint_idxs(self, tmp_path, data_format):
        if data_format == "ZYXC":
            pytest.skip("multi-timepoint subset indexing only applicable to TZYXC")

        n_t = 5
        data = _make_data(data_format, n_channels=2, n_timepoints=n_t)
        zarr_path = str(tmp_path / "img.zarr")
        self._create_store(zarr_path, data, data_format)

        subset_t = [0, 2, 4]
        mask = _make_mask(data_format, n_channels=1, n_timepoints=len(subset_t))
        update_zarr_data(
            image_path=zarr_path,
            data=mask,
            data_format=data_format,
            zarr_driver=ZARR_DRIVER,
            dtype=DTYPE,
            timepoint_idxs=subset_t,
            mode="append",
        )

        stored = _read_root(zarr_path)
        assert stored.shape[-1] == 3
        for i, t in enumerate(subset_t):
            np.testing.assert_array_equal(stored[t, ..., 2:], mask[i, ..., :].astype(np.uint16))

    @FORMAT_PARAMS
    def test_append_rejects_mask_channel_idxs(self, tmp_path, data_format):
        data = _make_data(data_format, n_channels=2)
        zarr_path = str(tmp_path / "img.zarr")
        self._create_store(zarr_path, data, data_format)

        mask = _make_mask(data_format, n_channels=1)
        with pytest.raises(ValueError, match="mode is 'append'"):
            update_zarr_data(
                image_path=zarr_path,
                data=mask,
                data_format=data_format,
                zarr_driver=ZARR_DRIVER,
                dtype=DTYPE,
                mask_channel_idxs=[2],
                timepoint_idxs=_tp(data_format),
                mode="append",
            )

    @FORMAT_PARAMS
    def test_append_spatial_mismatch_raises(self, tmp_path, data_format):
        data = _make_data(data_format, n_channels=2)
        zarr_path = str(tmp_path / "img.zarr")
        self._create_store(zarr_path, data, data_format)

        wrong_spatial = (8, 16, 32)  # X is different
        mask = _make_mask(data_format, n_channels=1, spatial=wrong_spatial)
        with pytest.raises(ValueError, match="different spatial dimensions"):
            update_zarr_data(
                image_path=zarr_path,
                data=mask,
                data_format=data_format,
                zarr_driver=ZARR_DRIVER,
                dtype=DTYPE,
                timepoint_idxs=_tp(data_format),
                mode="append",
            )

    def test_append_nonexistent_path_raises(self, tmp_path):
        mask = _make_mask("TZYXC", n_channels=1)
        with pytest.raises(FileNotFoundError):
            update_zarr_data(
                image_path=str(tmp_path / "nonexistent.zarr"),
                data=mask,
                data_format="TZYXC",
                zarr_driver=ZARR_DRIVER,
                dtype=DTYPE,
                mode="append",
            )


# ===========================================================================
# 4. update_zarr_data — overwrite mode (selective channel overwriting)
# ===========================================================================

class TestUpdateZarrOverwrite:
    def _create_store_with_masks(self, zarr_path: str, data_format: str, n_data_ch: int = 2, n_mask_ch: int = 2):
        """Create a store with data channels, then append mask channels."""
        data = _make_data(data_format, n_channels=n_data_ch)
        save_zarr_data(
            image_path=zarr_path,
            data=data,
            shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE,
            data_format=data_format,
            zarr_driver=ZARR_DRIVER,
            dtype=DTYPE,
        )
        mask = _make_mask(data_format, n_channels=n_mask_ch)
        update_zarr_data(
            image_path=zarr_path,
            data=mask,
            data_format=data_format,
            zarr_driver=ZARR_DRIVER,
            dtype=DTYPE,
            timepoint_idxs=_tp(data_format),
            mode="append",
        )
        return _to_disk(data, data_format).astype(np.uint16), _to_disk(mask, data_format).astype(np.uint16)

    @FORMAT_PARAMS
    def test_overwrite_last_mask_channel_only(self, tmp_path, data_format):
        """Overwrite channel -1 while preserving channel -2 and all data channels."""
        zarr_path = str(tmp_path / "img.zarr")
        orig_data, orig_mask = self._create_store_with_masks(zarr_path, data_format, n_data_ch=2, n_mask_ch=2)

        new_mask = (_make_mask(data_format, n_channels=1) + 10).astype(np.uint16)

        total_ch = orig_data.shape[-1] + orig_mask.shape[-1]  # 4
        update_zarr_data(
            image_path=zarr_path,
            data=new_mask,
            data_format=data_format,
            zarr_driver=ZARR_DRIVER,
            dtype=DTYPE,
            data_channel_idxs=[0, 1],
            mask_channel_idxs=[3],
            timepoint_idxs=_tp(data_format),
            mode="overwrite",
        )

        stored = _read_root(zarr_path)
        assert stored.shape[-1] == total_ch
        np.testing.assert_array_equal(stored[..., :2], orig_data, err_msg="Data channels corrupted")
        np.testing.assert_array_equal(stored[..., 2], orig_mask[..., 0], err_msg="First mask channel corrupted")
        np.testing.assert_array_equal(stored[..., 3], _to_disk(new_mask, data_format)[..., 0], err_msg="Last mask channel not updated")

    @FORMAT_PARAMS
    def test_overwrite_first_mask_channel_only(self, tmp_path, data_format):
        """Overwrite channel -2 while preserving channel -1 and all data channels."""
        zarr_path = str(tmp_path / "img.zarr")
        orig_data, orig_mask = self._create_store_with_masks(zarr_path, data_format, n_data_ch=2, n_mask_ch=2)

        new_mask = (_make_mask(data_format, n_channels=1) + 20).astype(np.uint16)

        update_zarr_data(
            image_path=zarr_path,
            data=new_mask,
            data_format=data_format,
            zarr_driver=ZARR_DRIVER,
            dtype=DTYPE,
            data_channel_idxs=[0, 1],
            mask_channel_idxs=[2],
            timepoint_idxs=_tp(data_format),
            mode="overwrite",
        )

        stored = _read_root(zarr_path)
        np.testing.assert_array_equal(stored[..., :2], orig_data, err_msg="Data channels corrupted")
        np.testing.assert_array_equal(stored[..., 2], _to_disk(new_mask, data_format)[..., 0], err_msg="First mask channel not updated")
        np.testing.assert_array_equal(stored[..., 3], orig_mask[..., 1], err_msg="Second mask channel corrupted")

    @FORMAT_PARAMS
    def test_overwrite_all_mask_channels(self, tmp_path, data_format):
        zarr_path = str(tmp_path / "img.zarr")
        orig_data, _ = self._create_store_with_masks(zarr_path, data_format, n_data_ch=2, n_mask_ch=2)

        new_masks = (_make_mask(data_format, n_channels=2) + 30).astype(np.uint16)
        update_zarr_data(
            image_path=zarr_path,
            data=new_masks,
            data_format=data_format,
            zarr_driver=ZARR_DRIVER,
            dtype=DTYPE,
            data_channel_idxs=[0, 1],
            mask_channel_idxs=[2, 3],
            timepoint_idxs=_tp(data_format),
            mode="overwrite",
        )

        stored = _read_root(zarr_path)
        np.testing.assert_array_equal(stored[..., :2], orig_data, err_msg="Data channels corrupted")
        np.testing.assert_array_equal(stored[..., 2:], _to_disk(new_masks, data_format), err_msg="Mask channels not updated")

    @FORMAT_PARAMS
    def test_overwrite_with_timepoint_idxs(self, tmp_path, data_format):
        if data_format == "ZYXC":
            pytest.skip("multi-timepoint subset indexing only applicable to TZYXC")

        n_t = 4
        zarr_path = str(tmp_path / "img.zarr")
        data = _make_data(data_format, n_channels=2, n_timepoints=n_t)
        save_zarr_data(
            image_path=zarr_path, data=data, shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, data_format=data_format,
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )
        mask_all = _make_mask(data_format, n_channels=1, n_timepoints=n_t)
        update_zarr_data(
            image_path=zarr_path, data=mask_all, data_format=data_format,
            zarr_driver=ZARR_DRIVER, dtype=DTYPE, mode="append",
        )

        subset_t = [1, 3]
        new_mask = (_make_mask(data_format, n_channels=1, n_timepoints=len(subset_t)) + 50).astype(np.uint16)
        update_zarr_data(
            image_path=zarr_path, data=new_mask, data_format=data_format,
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
            timepoint_idxs=subset_t,
            data_channel_idxs=[0, 1],
            mask_channel_idxs=[2],
            mode="overwrite",
        )

        stored = _read_root(zarr_path)
        for i, t in enumerate(subset_t):
            np.testing.assert_array_equal(stored[t, ..., 2], new_mask[i, ..., 0])
        np.testing.assert_array_equal(stored[0, ..., 2], mask_all[0, ..., 0].astype(np.uint16))

    @FORMAT_PARAMS
    def test_overwrite_guards_data_channels(self, tmp_path, data_format):
        """mask_channel_idxs overlapping data_channel_idxs should raise."""
        zarr_path = str(tmp_path / "img.zarr")
        self._create_store_with_masks(zarr_path, data_format, n_data_ch=2, n_mask_ch=2)

        new_mask = _make_mask(data_format, n_channels=1)
        with pytest.raises(ValueError, match="overwrite data channels"):
            update_zarr_data(
                image_path=zarr_path,
                data=new_mask,
                data_format=data_format,
                zarr_driver=ZARR_DRIVER,
                dtype=DTYPE,
                data_channel_idxs=[0, 1, 2],
                mask_channel_idxs=[1],
                timepoint_idxs=_tp(data_format),
                mode="overwrite",
            )

    @FORMAT_PARAMS
    def test_overwrite_requires_both_idx_args(self, tmp_path, data_format):
        zarr_path = str(tmp_path / "img.zarr")
        self._create_store_with_masks(zarr_path, data_format)

        mask = _make_mask(data_format, n_channels=1)
        with pytest.raises(ValueError):
            update_zarr_data(
                image_path=zarr_path, data=mask, data_format=data_format,
                zarr_driver=ZARR_DRIVER, dtype=DTYPE,
                data_channel_idxs=None, mask_channel_idxs=[2],
                timepoint_idxs=_tp(data_format),
                mode="overwrite",
            )
        with pytest.raises(ValueError):
            update_zarr_data(
                image_path=zarr_path, data=mask, data_format=data_format,
                zarr_driver=ZARR_DRIVER, dtype=DTYPE,
                data_channel_idxs=[0, 1], mask_channel_idxs=None,
                timepoint_idxs=_tp(data_format),
                mode="overwrite",
            )

    @FORMAT_PARAMS
    def test_overwrite_out_of_bounds_channel_raises(self, tmp_path, data_format):
        """mask_channel_idxs pointing beyond the store's channel count should raise."""
        zarr_path = str(tmp_path / "img.zarr")
        self._create_store_with_masks(zarr_path, data_format, n_data_ch=2, n_mask_ch=1)
        # Store has 3 channels total
        big_mask = _make_mask(data_format, n_channels=3)
        with pytest.raises(ValueError, match="out of bounds"):
            update_zarr_data(
                image_path=zarr_path, data=big_mask, data_format=data_format,
                zarr_driver=ZARR_DRIVER, dtype=DTYPE,
                data_channel_idxs=[0, 1], mask_channel_idxs=[2, 3, 4],
                timepoint_idxs=_tp(data_format),
                mode="overwrite",
            )

    @FORMAT_PARAMS
    def test_overwrite_channel_count_mismatch_raises(self, tmp_path, data_format):
        """len(mask_channel_idxs) != data channel count should raise."""
        zarr_path = str(tmp_path / "img.zarr")
        self._create_store_with_masks(zarr_path, data_format, n_data_ch=2, n_mask_ch=2)
        # Data has 1 channel but we specify 2 mask indices
        mask = _make_mask(data_format, n_channels=1)
        with pytest.raises(ValueError, match="must match"):
            update_zarr_data(
                image_path=zarr_path, data=mask, data_format=data_format,
                zarr_driver=ZARR_DRIVER, dtype=DTYPE,
                data_channel_idxs=[0, 1], mask_channel_idxs=[2, 3],
                timepoint_idxs=_tp(data_format),
                mode="overwrite",
            )


# ===========================================================================
# 5. annotation_exists
# ===========================================================================

class TestAnnotationExists:
    @FORMAT_PARAMS
    def test_returns_false_when_missing(self, tmp_path, data_format):
        data = _make_data(data_format)
        zarr_path = str(tmp_path / "img.zarr")
        save_zarr_data(
            image_path=zarr_path, data=data, shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, data_format=data_format,
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )
        assert annotation_exists(zarr_path, "mymodel", "semantic_masks", ZARR_DRIVER) is False

    @FORMAT_PARAMS
    def test_returns_true_after_creation(self, tmp_path, data_format):
        data = _make_data(data_format)
        zarr_path = str(tmp_path / "img.zarr")
        save_zarr_data(
            image_path=zarr_path, data=data, shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, data_format=data_format,
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )
        mask = _make_mask(data_format, n_channels=1)
        save_zarr_annotations(
            image_path=zarr_path, data=mask, source_name="mymodel",
            annotation_name="semantic_masks", data_format=data_format,
            save_mode="create", shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, timepoint_idxs=_tp(data_format),
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )
        assert annotation_exists(zarr_path, "mymodel", "semantic_masks", ZARR_DRIVER) is True


# ===========================================================================
# 6. save_zarr_annotations — create & overwrite
# ===========================================================================

class TestSaveZarrAnnotations:
    @FORMAT_PARAMS
    def test_create_annotation_array(self, tmp_path, data_format):
        data = _make_data(data_format)
        zarr_path = str(tmp_path / "img.zarr")
        save_zarr_data(
            image_path=zarr_path, data=data, shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, data_format=data_format,
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )

        mask = _make_mask(data_format, n_channels=1)
        save_zarr_annotations(
            image_path=zarr_path, data=mask, source_name="modelA",
            annotation_name="instance_masks", data_format=data_format,
            save_mode="create", shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, timepoint_idxs=_tp(data_format),
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )

        stored = _read_annotation(zarr_path, "modelA", "instance_masks")
        np.testing.assert_array_equal(stored, _to_disk(mask, data_format).astype(np.uint16))

    @FORMAT_PARAMS
    def test_create_duplicate_raises(self, tmp_path, data_format):
        data = _make_data(data_format)
        zarr_path = str(tmp_path / "img.zarr")
        save_zarr_data(
            image_path=zarr_path, data=data, shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, data_format=data_format,
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )

        mask = _make_mask(data_format, n_channels=1)
        save_zarr_annotations(
            image_path=zarr_path, data=mask, source_name="modelA",
            annotation_name="instance_masks", data_format=data_format,
            save_mode="create", shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, timepoint_idxs=_tp(data_format),
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )

        with pytest.raises(ValueError, match="already exists"):
            save_zarr_annotations(
                image_path=zarr_path, data=mask, source_name="modelA",
                annotation_name="instance_masks", data_format=data_format,
                save_mode="create", shard_spatial_shape=SHARD_SPATIAL_SHAPE,
                chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, timepoint_idxs=_tp(data_format),
                zarr_driver=ZARR_DRIVER, dtype=DTYPE,
            )

    @FORMAT_PARAMS
    def test_overwrite_annotation_array(self, tmp_path, data_format):
        data = _make_data(data_format)
        zarr_path = str(tmp_path / "img.zarr")
        save_zarr_data(
            image_path=zarr_path, data=data, shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, data_format=data_format,
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )

        mask1 = _make_mask(data_format, n_channels=1)
        save_zarr_annotations(
            image_path=zarr_path, data=mask1, source_name="modelA",
            annotation_name="semantic_masks", data_format=data_format,
            save_mode="create", shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, timepoint_idxs=_tp(data_format),
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )

        mask2 = (_make_mask(data_format, n_channels=1) + 7).astype(np.uint16)
        save_zarr_annotations(
            image_path=zarr_path, data=mask2, source_name="modelA",
            annotation_name="semantic_masks", data_format=data_format,
            save_mode="overwrite", timepoint_idxs=_tp(data_format),
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )

        stored = _read_annotation(zarr_path, "modelA", "semantic_masks")
        np.testing.assert_array_equal(stored, _to_disk(mask2, data_format).astype(np.uint16))

    @FORMAT_PARAMS
    def test_overwrite_nonexistent_creates(self, tmp_path, data_format):
        """When save_mode='overwrite' and the annotation does not yet exist, create it."""
        data = _make_data(data_format)
        zarr_path = str(tmp_path / "img.zarr")
        save_zarr_data(
            image_path=zarr_path, data=data, shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, data_format=data_format,
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )

        mask = _make_mask(data_format, n_channels=1)
        save_zarr_annotations(
            image_path=zarr_path, data=mask, source_name="modelA",
            annotation_name="semantic_masks", data_format=data_format,
            save_mode="overwrite", shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, timepoint_idxs=_tp(data_format),
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )

        stored = _read_annotation(zarr_path, "modelA", "semantic_masks")
        np.testing.assert_array_equal(stored, _to_disk(mask, data_format).astype(np.uint16))

    @FORMAT_PARAMS
    def test_overwrite_with_timepoint_idxs(self, tmp_path, data_format):
        if data_format == "ZYXC":
            pytest.skip("multi-timepoint subset indexing only applicable to TZYXC")

        n_t = 4
        data = _make_data(data_format, n_channels=2, n_timepoints=n_t)
        zarr_path = str(tmp_path / "img.zarr")
        save_zarr_data(
            image_path=zarr_path, data=data, shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, data_format=data_format,
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )

        mask_all = _make_mask(data_format, n_channels=1, n_timepoints=n_t)
        save_zarr_annotations(
            image_path=zarr_path, data=mask_all, source_name="modelA",
            annotation_name="instance_masks", data_format=data_format,
            save_mode="create", shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )

        subset_t = [1, 3]
        mask_sub = (_make_mask(data_format, n_channels=1, n_timepoints=len(subset_t)) + 42).astype(np.uint16)
        save_zarr_annotations(
            image_path=zarr_path, data=mask_sub, source_name="modelA",
            annotation_name="instance_masks", data_format=data_format,
            save_mode="overwrite", timepoint_idxs=subset_t,
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )

        stored = _read_annotation(zarr_path, "modelA", "instance_masks")
        for i, t in enumerate(subset_t):
            np.testing.assert_array_equal(stored[t], mask_sub[i].astype(np.uint16))
        np.testing.assert_array_equal(stored[0], mask_all[0].astype(np.uint16))

    def test_invalid_source_name_raises(self, tmp_path):
        zarr_path = str(tmp_path / "img.zarr")
        data = _make_data("TZYXC")
        save_zarr_data(
            image_path=zarr_path, data=data, shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, data_format="TZYXC",
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )
        mask = _make_mask("TZYXC", n_channels=1)
        with pytest.raises(ValueError, match="Invalid source name"):
            save_zarr_annotations(
                image_path=zarr_path, data=mask, source_name="bad/name",
                annotation_name="masks", data_format="TZYXC",
                save_mode="create", shard_spatial_shape=SHARD_SPATIAL_SHAPE,
                chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, zarr_driver=ZARR_DRIVER, dtype=DTYPE,
            )

    def test_invalid_label_name_raises(self, tmp_path):
        zarr_path = str(tmp_path / "img.zarr")
        data = _make_data("TZYXC")
        save_zarr_data(
            image_path=zarr_path, data=data, shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, data_format="TZYXC",
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )
        mask = _make_mask("TZYXC", n_channels=1)
        with pytest.raises(ValueError, match="Invalid annotation name"):
            save_zarr_annotations(
                image_path=zarr_path, data=mask, source_name="modelA",
                annotation_name="bad/label", data_format="TZYXC",
                save_mode="create", shard_spatial_shape=SHARD_SPATIAL_SHAPE,
                chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, zarr_driver=ZARR_DRIVER, dtype=DTYPE,
            )

    @FORMAT_PARAMS
    def test_multiple_models_independent(self, tmp_path, data_format):
        """Two models writing to the same zarr should produce independent annotation arrays."""
        data = _make_data(data_format)
        zarr_path = str(tmp_path / "img.zarr")
        save_zarr_data(
            image_path=zarr_path, data=data, shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, data_format=data_format,
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )

        mask_a = _make_mask(data_format, n_channels=1)
        mask_b = (_make_mask(data_format, n_channels=1) + 3).astype(np.uint16)

        save_zarr_annotations(
            image_path=zarr_path, data=mask_a, source_name="modelA",
            annotation_name="instance_masks", data_format=data_format,
            save_mode="create", shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, timepoint_idxs=_tp(data_format),
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )
        save_zarr_annotations(
            image_path=zarr_path, data=mask_b, source_name="modelB",
            annotation_name="instance_masks", data_format=data_format,
            save_mode="create", shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, timepoint_idxs=_tp(data_format),
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )

        stored_a = _read_annotation(zarr_path, "modelA", "instance_masks")
        stored_b = _read_annotation(zarr_path, "modelB", "instance_masks")
        np.testing.assert_array_equal(stored_a, _to_disk(mask_a, data_format).astype(np.uint16))
        np.testing.assert_array_equal(stored_b, _to_disk(mask_b, data_format).astype(np.uint16))


# ===========================================================================
# 7. save_masks (high-level orchestration)
# ===========================================================================

class TestSaveMasks:
    @FORMAT_PARAMS
    @pytest.mark.parametrize(
        "task, mask_root_name",
        [
            ("semantic_segmentation", "semantic_masks"),
            ("instance_segmentation", "instance_masks"),
        ],
    )
    def test_save_masks_append_creates_root_and_annotation(
        self, tmp_path, data_format, task, mask_root_name
    ):
        data = _make_data(data_format, n_channels=2)
        zarr_path = str(tmp_path / "img.zarr")
        save_zarr_data(
            image_path=zarr_path,
            data=data,
            shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE,
            data_format=data_format,
            zarr_driver=ZARR_DRIVER,
            dtype=DTYPE,
            channel_names={0: "d0", 1: "d1"},
        )

        mask = _make_mask(data_format, n_channels=1)
        model_name = "my_model"

        save_masks(
            image_path=zarr_path,
            masks=mask,
            annotation_name=mask_root_name,
            existing_channel_names={0: "d0", 1: "d1"},
            data_format=data_format,
            model_name=model_name,
            save_mode="create",
            zarr_driver=ZARR_DRIVER,
            dtype=DTYPE,
            shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE,
            timepoint_idxs=_tp(data_format),
        )

        root = _read_root(zarr_path)
        assert root.shape[-1] == 3  # 2 data + 1 appended
        np.testing.assert_array_equal(root[..., :2], _to_disk(data, data_format).astype(np.uint16))
        assert read_channel_names(zarr_path)[2] == mask_root_name
        annotation = _read_annotation(zarr_path, model_name, mask_root_name)
        np.testing.assert_array_equal(annotation, _to_disk(mask, data_format).astype(np.uint16))

    @FORMAT_PARAMS
    def test_save_masks_overwrite_first_write_appends_like_create(self, tmp_path, data_format):
        """overwrite with no root channel named annotation_name uses append pipeline (data-only names ok)."""
        data = _make_data(data_format, n_channels=2)
        zarr_path = str(tmp_path / "img.zarr")
        save_zarr_data(
            image_path=zarr_path,
            data=data,
            shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE,
            data_format=data_format,
            zarr_driver=ZARR_DRIVER,
            dtype=DTYPE,
            channel_names={0: "d0", 1: "d1"},
        )

        mask = _make_mask(data_format, n_channels=1)
        model_name = "model_ovw"
        mask_root_name = "semantic_masks"

        save_masks(
            image_path=zarr_path,
            masks=mask,
            annotation_name=mask_root_name,
            existing_channel_names={0: "d0", 1: "d1"},
            data_format=data_format,
            model_name=model_name,
            save_mode="overwrite",
            zarr_driver=ZARR_DRIVER,
            dtype=DTYPE,
            shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE,
            timepoint_idxs=_tp(data_format),
        )

        root = _read_root(zarr_path)
        assert root.shape[-1] == 3
        assert read_channel_names(zarr_path)[2] == mask_root_name
        annotation = _read_annotation(zarr_path, model_name, mask_root_name)
        np.testing.assert_array_equal(annotation, _to_disk(mask, data_format).astype(np.uint16))

    @FORMAT_PARAMS
    def test_save_masks_overwrite_rerun_same_model(self, tmp_path, data_format):
        """Re-running the same model overwrites the annotation array and updates root."""
        data = _make_data(data_format, n_channels=2)
        zarr_path = str(tmp_path / "img.zarr")
        save_zarr_data(
            image_path=zarr_path,
            data=data,
            shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE,
            data_format=data_format,
            zarr_driver=ZARR_DRIVER,
            dtype=DTYPE,
            channel_names={0: "d0", 1: "d1"},
        )

        mask1 = _make_mask(data_format, n_channels=1)
        save_masks(
            image_path=zarr_path,
            masks=mask1,
            annotation_name="semantic_masks",
            existing_channel_names={0: "d0", 1: "d1"},
            data_format=data_format,
            model_name="modelX",
            save_mode="create",
            zarr_driver=ZARR_DRIVER,
            dtype=DTYPE,
            shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE,
            timepoint_idxs=_tp(data_format),
        )

        mask2 = (_make_mask(data_format, n_channels=1) + 9).astype(np.uint16)
        save_masks(
            image_path=zarr_path,
            masks=mask2,
            annotation_name="semantic_masks",
            existing_channel_names={0: "d0", 1: "d1", 2: "semantic_masks"},
            data_format=data_format,
            model_name="modelX",
            save_mode="overwrite",
            zarr_driver=ZARR_DRIVER,
            dtype=DTYPE,
            shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE,
            timepoint_idxs=_tp(data_format),
        )

        root = _read_root(zarr_path)
        np.testing.assert_array_equal(
            root[..., :2],
            _to_disk(data, data_format).astype(np.uint16),
            err_msg="Data channels corrupted by save_masks overwrite",
        )

        annotation = _read_annotation(zarr_path, "modelX", "semantic_masks")
        np.testing.assert_array_equal(annotation, _to_disk(mask2, data_format).astype(np.uint16))


# ===========================================================================
# 8. normalize_idxs
# ===========================================================================

class TestNormalizeIdxs:
    def test_positive_passthrough(self):
        assert normalize_idxs([0, 1, 2], 5) == [0, 1, 2]

    def test_negative_conversion(self):
        assert normalize_idxs([-1, -2], 5) == [4, 3]

    def test_float_integer_accepted(self):
        assert normalize_idxs([1.0, 2.0], 5) == [1, 2]

    def test_float_non_integer_raises(self):
        with pytest.raises(ValueError, match="not integer-valued"):
            normalize_idxs([1.5], 5)

    def test_out_of_bounds_raises(self):
        with pytest.raises(ValueError, match="out of bounds"):
            normalize_idxs([5], 5)

    def test_negative_out_of_bounds_raises(self):
        with pytest.raises(ValueError, match="out of bounds"):
            normalize_idxs([-6], 5)


# ===========================================================================
# 10. Edge cases & data integrity
# ===========================================================================

class TestDataIntegrity:
    @FORMAT_PARAMS
    def test_root_data_untouched_after_append(self, tmp_path, data_format):
        """Core invariant: appending masks must never alter existing data channels."""
        data = _make_data(data_format, n_channels=3)
        zarr_path = str(tmp_path / "img.zarr")
        save_zarr_data(
            image_path=zarr_path, data=data, shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, data_format=data_format,
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )
        original_data = _read_root(zarr_path).copy()

        for i in range(3):
            mask = _make_mask(data_format, n_channels=1)
            update_zarr_data(
                image_path=zarr_path, data=mask, data_format=data_format,
                zarr_driver=ZARR_DRIVER, dtype=DTYPE, timepoint_idxs=_tp(data_format),
                mode="append",
            )

        stored = _read_root(zarr_path)
        assert stored.shape[-1] == 6  # 3 data + 3 masks
        np.testing.assert_array_equal(
            stored[..., :3], original_data,
            err_msg="Data channels were corrupted after multiple appends",
        )

    @FORMAT_PARAMS
    def test_root_data_untouched_after_overwrite(self, tmp_path, data_format):
        """Core invariant: overwriting masks must never alter data channels."""
        data = _make_data(data_format, n_channels=2)
        zarr_path = str(tmp_path / "img.zarr")
        save_zarr_data(
            image_path=zarr_path, data=data, shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, data_format=data_format,
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )
        mask = _make_mask(data_format, n_channels=2)
        update_zarr_data(
            image_path=zarr_path, data=mask, data_format=data_format,
            zarr_driver=ZARR_DRIVER, dtype=DTYPE, timepoint_idxs=_tp(data_format),
            mode="append",
        )
        original_data = _read_root(zarr_path)[..., :2].copy()

        for _ in range(5):
            new_mask = _make_mask(data_format, n_channels=2)
            update_zarr_data(
                image_path=zarr_path, data=new_mask, data_format=data_format,
                zarr_driver=ZARR_DRIVER, dtype=DTYPE,
                data_channel_idxs=[0, 1], mask_channel_idxs=[2, 3],
                timepoint_idxs=_tp(data_format),
                mode="overwrite",
            )

        stored = _read_root(zarr_path)
        np.testing.assert_array_equal(
            stored[..., :2], original_data,
            err_msg="Data channels were corrupted after multiple overwrites",
        )

    @FORMAT_PARAMS
    def test_annotation_creation_does_not_affect_root(self, tmp_path, data_format):
        """Creating annotation groups must not change the root array at all."""
        data = _make_data(data_format, n_channels=2)
        zarr_path = str(tmp_path / "img.zarr")
        save_zarr_data(
            image_path=zarr_path, data=data, shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, data_format=data_format,
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )
        original = _read_root(zarr_path).copy()

        mask = _make_mask(data_format, n_channels=1)
        save_zarr_annotations(
            image_path=zarr_path, data=mask, source_name="modelA",
            annotation_name="instance_masks", data_format=data_format,
            save_mode="create", shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, timepoint_idxs=_tp(data_format),
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )

        after = _read_root(zarr_path)
        np.testing.assert_array_equal(after, original, err_msg="Root array corrupted by annotation creation")

    @FORMAT_PARAMS
    def test_dimension_mismatch_raises(self, tmp_path, data_format):
        data = _make_data(data_format, n_channels=2)
        zarr_path = str(tmp_path / "img.zarr")
        save_zarr_data(
            image_path=zarr_path, data=data, shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, data_format=data_format,
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )

        wrong_ndim = np.zeros((8, 16, 16))
        with pytest.raises(ValueError, match="same number of dimensions"):
            update_zarr_data(
                image_path=zarr_path, data=wrong_ndim, data_format=data_format,
                zarr_driver=ZARR_DRIVER, dtype=DTYPE, mode="append",
            )

    def test_invalid_mode_raises(self, tmp_path):
        data = _make_data("TZYXC")
        zarr_path = str(tmp_path / "img.zarr")
        save_zarr_data(
            image_path=zarr_path, data=data, shard_spatial_shape=SHARD_SPATIAL_SHAPE,
            chunk_spatial_shape=CHUNK_SPATIAL_SHAPE, data_format="TZYXC",
            zarr_driver=ZARR_DRIVER, dtype=DTYPE,
        )
        mask = _make_mask("TZYXC", n_channels=1)
        with pytest.raises(ValueError, match="must be specified for overwriting"):
            update_zarr_data(
                image_path=zarr_path, data=mask, data_format="TZYXC",
                zarr_driver=ZARR_DRIVER, dtype=DTYPE, mode="overwrite",
            )
