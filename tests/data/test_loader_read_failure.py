"""LoaderActor.read_failure_policy: an undecodable zarr chunk must not abort the run.

Builds a real sharded zarr3 array (same codec stack as production), corrupts one
shard on disk, and drives LoaderActor.__call__ without Ray (buffer actor / shared
memory stubbed). The failed sample region is zero-filled as ONE unit -- image and
mask channels come from the same tensorstore read -- healthy samples are untouched,
a warning is emitted once per (tile, region), and the counter increments per event.
"""

import logging
import os

import numpy as np
import pytest
import tensorstore as ts

from cell_observatory_platform.data.datasets import pretrain_dataset_ray as pdr
from cell_observatory_platform.data.datasets.pretrain_dataset_ray import LoaderActor
from cell_observatory_platform.data.io import _make_write_zarr_spec

T, Z, Y, X, C = 2, 8, 8, 8, 2
SHARD = (1, 4, 4, 4, C)
CHUNK = (1, 2, 2, 2, C)
SAMPLE = (1, 4, 4, 4)  # t, z, y, x extents per sample
B = 2


@pytest.fixture(autouse=True)
def _no_ray(monkeypatch):
    # No Ray in this test: ray.get on the stubbed buffer actor is identity, and
    # LoaderActor.__del__ must not auto-init Ray via ray.get_actor.
    monkeypatch.setattr(pdr.ray, "get", lambda x: x)
    monkeypatch.setattr(LoaderActor, "__del__", lambda self: None)


@pytest.fixture
def zarr_with_bad_shard(tmp_path):
    path = str(tmp_path / "tile.zarr")
    spec = _make_write_zarr_spec((T, Z, Y, X, C), "zarr3", path, CHUNK, shard_shape=SHARD, dtype="uint16")
    arr = ts.open(spec).result()
    data = np.random.default_rng(0).integers(1, 1000, size=(T, Z, Y, X, C), dtype=np.uint16)
    arr.write(data).result()
    # Corrupt the shard holding t=0, z/y/x block 0 (sample 0's region); t=1 stays healthy.
    bad = os.path.join(path, "c", "0", "0", "0", "0", "0")
    assert os.path.exists(bad), os.listdir(os.path.join(path, "c"))
    size = os.path.getsize(bad)
    with open(bad, "wb") as f:
        f.write(np.random.default_rng(1).bytes(size))
    return path, data


class _FakeBufferActor:
    class _Free:
        @staticmethod
        def remote():
            return {"slot": 0, "name": "fake"}

    get_free = _Free


class _FakeShm:
    def __init__(self, nbytes):
        self.buf = bytearray(nbytes)


def _make_loader(policy: str) -> LoaderActor:
    la = LoaderActor.__new__(LoaderActor)  # skip Ray/NUMA wiring in __init__
    la.dim, la.input_format, la.pad_mode, la.last_batch_policy, la.save_mode = 4, "TZYXC", "zero", "drop", None
    la.node_id = la.local_rank = la.global_rank = 0
    la.selected_channel_localizations = None
    la.input_layout, la.batch_size = "TZYXC", B
    la.buffer_dtype = np.uint16
    la._handles, la.ctx, la.with_batched_api = {}, ts.Context(), True
    la.batch_shape = (B, *SAMPLE, C)
    la.slot_bytes = int(np.prod(la.batch_shape)) * 2
    la._shm = _FakeShm(la.slot_bytes)
    la.buffer_actor = _FakeBufferActor()
    la.read_failure_policy = policy
    la.read_failure_count = 0
    la._read_failure_warned = set()
    return la


def _batch(path):
    n = B
    return {
        "storage_root": np.array([os.path.dirname(path)] * n, dtype=object),
        "tile_relative_path": np.array([os.path.basename(path)] * n, dtype=object),
        "time_start": np.array([0, 1]),  # sample 0 hits the corrupt shard, sample 1 does not
        "time_size": np.array([1, 1]),
        "z_start": np.zeros(n, dtype=int),
        "y_start": np.zeros(n, dtype=int),
        "x_start": np.zeros(n, dtype=int),
        "z_size": np.full(n, 4),
        "y_size": np.full(n, 4),
        "x_size": np.full(n, 4),
        "channel_size": np.full(n, C),
        "channel_idx": np.array([[0, 1]] * n, dtype=object),
        "channel_type": np.array([["signal", "mask"]] * n, dtype=object),
        "localization": np.array([["membrane", None]] * n, dtype=object),
        "annotation_type": np.array([[None, "instance"]] * n, dtype=object),
    }


def _dst(la):
    return np.ndarray(la.batch_shape, dtype=la.buffer_dtype, buffer=la._shm.buf, offset=0)


def test_zero_fill_policy_zeroes_failed_sample_and_warns_once(zarr_with_bad_shard, caplog):
    path, data = zarr_with_bad_shard
    la = _make_loader("zero_fill")
    dst = _dst(la)
    dst.fill(7)  # stale content from a previous batch must not survive

    with caplog.at_level(logging.WARNING, logger=pdr.logger.name):
        out = la(_batch(path))

    # failed sample: whole region (image AND mask channel) is zero
    assert not dst[0].any()
    # healthy sample: exact on-disk values, incl. mask channel
    np.testing.assert_array_equal(dst[1], data[1:2, :4, :4, :4, :])
    assert la.read_failure_count == 1
    assert la.get_read_failure_count() == 1
    assert out["valid_mask"].all()

    warnings = [r for r in caplog.records if "[LOADER] unreadable region" in r.getMessage()]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert path in msg and "t[0:1] z[0:4] y[0:4] x[0:4]" in msg and "zero-filled" in msg

    # same (tile, region) again: counted, but warning is rate-limited to once
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=pdr.logger.name):
        la(_batch(path))
    assert la.read_failure_count == 2
    assert not [r for r in caplog.records if "[LOADER] unreadable region" in r.getMessage()]
    assert not dst[0].any()


def test_raise_policy_propagates(zarr_with_bad_shard):
    path, _ = zarr_with_bad_shard
    la = _make_loader("raise")
    with pytest.raises(Exception):
        la(_batch(path))
    assert la.read_failure_count == 0


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        # validation happens before any Ray/buffer wiring in __init__
        la = LoaderActor.__new__(LoaderActor)
        LoaderActor.__init__(
            la, dim=4, input_format="TZYXC", node_id=0, local_rank=0, global_rank=0, numa_node=0,
            batch_size=1, input_layout="TZYXC", context_spec={}, pin_numa_node=False,
            read_failure_policy="bogus",
        )
