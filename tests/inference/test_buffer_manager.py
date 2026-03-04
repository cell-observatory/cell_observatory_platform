"""Unit tests for BufferManager (alloc, free, metrics, pool stubs)."""

from unittest.mock import MagicMock, patch

import pytest

from cell_observatory_platform.inference.buffer_manager import BufferManager, SlotHandle


@pytest.fixture
def mock_set_output_buffers():
    """Mock set_output_buffers to avoid Ray/SharedMemory in tests."""
    with patch(
        "cell_observatory_platform.inference.buffer_manager.set_output_buffers"
    ) as m:
        mock_actor = MagicMock()
        mock_actor.get_free.remote.return_value = {
            "slot": 0,
            "name": "test_shm",
            "slot_bytes": 48,
            "batch_shape": [2, 24],
            "dtype": "uint8",
            "capacity": 8,
        }
        mock_actor.put_free.remote.return_value = True
        m.return_value = (mock_actor, {"name": "test_shm", "slot_bytes": 48})
        yield m, mock_actor


def test_buffer_manager_alloc_save_output_returns_handle(mock_set_output_buffers):
    """alloc('save_output') returns SlotHandle with correct fields."""
    _, mock_actor = mock_set_output_buffers
    with patch("cell_observatory_platform.inference.buffer_manager.ray") as mock_ray:
        mock_ray.get.return_value = {
            "slot": 0,
            "name": "test_shm",
            "slot_bytes": 48,
            "batch_shape": [2, 24],
            "dtype": "uint8",
            "capacity": 8,
        }

        bm = BufferManager(
            local_rank=0,
            global_rank=0,
            node_id=0,
            numa_node=0,
            batch_size=2,
            slot_bytes_total=24,
            buffer_capacity=8,
        )
        handle = bm.alloc("save_output")

        assert isinstance(handle, SlotHandle)
        assert handle.pool_class == "save_output"
        assert handle.slot_idx == 0
        assert handle.shm_name == "test_shm"
        assert handle.slot_bytes == 48
        assert handle.batch_size == 2
        assert handle.batch_shape == (2, 24)


def test_buffer_manager_free_calls_put_free(mock_set_output_buffers):
    """free('save_output', handle) calls buffer actor put_free."""
    _, mock_actor = mock_set_output_buffers
    with patch("cell_observatory_platform.inference.buffer_manager.ray") as mock_ray:
        mock_ray.get.side_effect = [
            {"slot": 0, "name": "s", "slot_bytes": 48, "batch_shape": [2, 24], "dtype": "uint8", "capacity": 8},
            True,
        ]

        bm = BufferManager(
            local_rank=0,
            global_rank=0,
            node_id=0,
            numa_node=0,
            batch_size=2,
            slot_bytes_total=24,
            buffer_capacity=8,
        )
        handle = bm.alloc("save_output")
        bm.free("save_output", handle)

        mock_actor.put_free.remote.assert_called_once_with(0)


def test_buffer_manager_viz_output_non_blocking_returns_none(mock_set_output_buffers):
    """alloc('viz_output', block=False) returns None (stub pool)."""
    bm = BufferManager(
        local_rank=0,
        global_rank=0,
        node_id=0,
        numa_node=0,
        batch_size=2,
        slot_bytes_total=24,
        buffer_capacity=8,
    )
    handle = bm.alloc("viz_output", block=False)
    assert handle is None


def test_buffer_manager_viz_output_blocking_raises(mock_set_output_buffers):
    """alloc('viz_output', block=True) raises NotImplementedError."""
    bm = BufferManager(
        local_rank=0,
        global_rank=0,
        node_id=0,
        numa_node=0,
        batch_size=2,
        slot_bytes_total=24,
        buffer_capacity=8,
    )
    with pytest.raises(NotImplementedError, match="viz_output blocking alloc not supported"):
        bm.alloc("viz_output", block=True)


def test_buffer_manager_preproc_postproc_raise(mock_set_output_buffers):
    """alloc('preproc_input') and alloc('postproc_input') raise NotImplementedError."""
    bm = BufferManager(
        local_rank=0,
        global_rank=0,
        node_id=0,
        numa_node=0,
        batch_size=2,
        slot_bytes_total=24,
        buffer_capacity=8,
    )
    with pytest.raises(NotImplementedError, match="preproc_input"):
        bm.alloc("preproc_input")
    with pytest.raises(NotImplementedError, match="postproc_input"):
        bm.alloc("postproc_input")


def test_buffer_manager_unknown_pool_raises(mock_set_output_buffers):
    """alloc('unknown') raises ValueError."""
    bm = BufferManager(
        local_rank=0,
        global_rank=0,
        node_id=0,
        numa_node=0,
        batch_size=2,
        slot_bytes_total=24,
        buffer_capacity=8,
    )
    with pytest.raises(ValueError, match="Unknown pool_class"):
        bm.alloc("unknown")


def test_buffer_manager_free_pool_class_mismatch_raises(mock_set_output_buffers):
    """free with mismatched pool_class raises ValueError."""
    _, mock_actor = mock_set_output_buffers
    with patch("cell_observatory_platform.inference.buffer_manager.ray") as mock_ray:
        mock_ray.get.return_value = {
            "slot": 0,
            "name": "s",
            "slot_bytes": 48,
            "batch_shape": [2, 24],
            "dtype": "uint8",
            "capacity": 8,
        }

        bm = BufferManager(
            local_rank=0,
            global_rank=0,
            node_id=0,
            numa_node=0,
            batch_size=2,
            slot_bytes_total=24,
            buffer_capacity=8,
        )
        handle = SlotHandle(
            pool_class="viz_output",
            slot_idx=0,
            shm_name="x",
            slot_bytes=48,
            batch_size=2,
            batch_shape=(2, 24),
        )
        with pytest.raises(ValueError, match="handle pool_class viz_output != free pool_class save_output"):
            bm.free("save_output", handle)


def test_buffer_manager_get_metrics(mock_set_output_buffers):
    """get_metrics returns dict with save_output and viz_output stats."""
    _, mock_actor = mock_set_output_buffers
    with patch("cell_observatory_platform.inference.buffer_manager.ray") as mock_ray:
        mock_ray.get.side_effect = [
            {"slot": 0, "name": "s", "slot_bytes": 48, "batch_shape": [2, 24], "dtype": "uint8", "capacity": 8},
            True,
        ]

        bm = BufferManager(
            local_rank=0,
            global_rank=0,
            node_id=0,
            numa_node=0,
            batch_size=2,
            slot_bytes_total=24,
            buffer_capacity=8,
        )
        bm.alloc("save_output")
        bm.free("save_output", SlotHandle("save_output", 0, "s", 48, 2, (2, 24)))
        bm.alloc("viz_output", block=False)

        metrics = bm.get_metrics()
        assert "save_output" in metrics
        assert metrics["save_output"]["alloc_count"] == 1
        assert "high_water_slots" in metrics["save_output"]
        assert "viz_output" in metrics
        assert metrics["viz_output"]["drops"] == 1


def test_buffer_manager_output_buffer_cfg(mock_set_output_buffers):
    """output_buffer_cfg property returns config dict."""
    bm = BufferManager(
        local_rank=0,
        global_rank=0,
        node_id=0,
        numa_node=0,
        batch_size=2,
        slot_bytes_total=24,
        buffer_capacity=8,
    )
    cfg = bm.output_buffer_cfg
    assert cfg["name"] == "test_shm"
    assert cfg["slot_bytes"] == 48
