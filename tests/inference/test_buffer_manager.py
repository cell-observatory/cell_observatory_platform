"""Unit tests for BufferManager and supporting components in data/datasets/buffers.py.

Tests are organised in three tiers:
  Tier 1 -- Pure utility functions (no Ray, no CUDA)
  Tier 2 -- HostMemoryBuffer Ray actor (requires Ray, no CUDA)
  Tier 3 -- BufferManager integration (requires Ray, mocks CUDA)
"""

import pickle
import time
import uuid
from multiprocessing import shared_memory

import numpy as np
import pytest
import ray

from tests.ray_init_helpers import init_ray_like_training

from cell_observatory_platform.data.datasets.buffers import (
    BufferManager,
    HostMemoryBuffer,
    get_buffer_name,
    get_slot_bytes,
    init_output_memory_pools,
    parse_buffer_name,
    set_buffers,
    slot_info_to_view,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ray_ctx():
    """Start a local Ray cluster via ``init_ray_like_training`` (RuntimeEnv with py_modules + env; no full-repo working_dir by default)."""
    init_ray_like_training(num_cpus=4, num_gpus=0)
    yield
    ray.shutdown()


@pytest.fixture
def ray_node_id(ray_ctx):
    """Return the hex node-ID of the single-node test cluster."""
    return ray.nodes()[0]["NodeID"]


def _uid() -> str:
    return uuid.uuid4().hex[:8]


@pytest.fixture
def unique_suffix():
    """Unique suffix for actor / pool names to avoid cross-test collisions."""
    return _uid()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_host_buffer(
    name: str,
    *,
    capacity: int = 4,
    input_shape: tuple = (4, 4),
    batch_size: int = 2,
    dtype: str = "uint16",
    namespace: str = "test_buffers",
):
    """Create a HostMemoryBuffer actor with NUMA binding disabled."""
    return HostMemoryBuffer.options(
        name=name,
        namespace=namespace,
        lifetime="detached",
        max_concurrency=10,
    ).remote(
        name=name,
        capacity=capacity,
        input_shape=input_shape,
        batch_size=batch_size,
        dtype=dtype,
        pin_numa_node=False,
        numa_node=0,
    )


def _kill_safe(handle):
    """Kill a Ray actor, suppressing errors."""
    try:
        ray.kill(handle)
    except Exception:
        pass


def _make_buffer_manager(
    ray_node_id,
    *,
    global_rank: int = 0,
    local_rank: int = 0,
    numa_node: int = 0,
    budget_gb: float = 1.0,
    max_concurrent: int = 10,
    safety_margin: float = 0.0,
):
    """Create a BufferManager wired to the test Ray cluster."""
    return BufferManager(
        local_rank=local_rank,
        global_rank=global_rank,
        node_id=ray_node_id,
        numa_node=numa_node,
        rank_memory_budget_gb=budget_gb,
        max_concurrent_calls=max_concurrent,
        safety_margin=safety_margin,
    )


def _make_shm(n_slots: int, shape: tuple, dtype):
    """Allocate a SharedMemory segment sized for *n_slots* arrays of *shape*/*dtype*."""
    slot_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    total = slot_bytes * n_slots
    shm = shared_memory.SharedMemory(create=True, size=total)
    return shm, slot_bytes


# ===========================================================================
# Tier 1: Pure Utility Functions (no Ray, no CUDA)
# ===========================================================================


class TestParseBufferName:
    def test_valid(self):
        result = parse_buffer_name(
            "host_pinned_shm_buffer_foo_save_numa_0_rank_3"
        )
        assert result == {
            "pool_name": "foo_save",
            "numa_node": "0",
            "global_rank": "3",
        }

    def test_pool_name_with_underscores(self):
        result = parse_buffer_name(
            "host_pinned_shm_buffer_my_long_pool_numa_2_rank_10"
        )
        assert result["pool_name"] == "my_long_pool"
        assert result["numa_node"] == "2"
        assert result["global_rank"] == "10"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid buffer name"):
            parse_buffer_name("garbage_string")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Invalid buffer name"):
            parse_buffer_name("")


class TestGetBufferName:
    def test_format(self):
        assert get_buffer_name("my_pool", 1, 5) == (
            "host_pinned_shm_buffer_my_pool_numa_1_rank_5"
        )

    def test_roundtrip_with_parse(self):
        pool, numa, rank = "seg_save", 3, 7
        name = get_buffer_name(pool, numa, rank)
        parsed = parse_buffer_name(name)
        assert parsed["pool_name"] == pool
        assert parsed["numa_node"] == str(numa)
        assert parsed["global_rank"] == str(rank)


class TestGetSlotBytes:
    def test_uint16_3d(self):
        assert get_slot_bytes((8, 8, 8), "uint16") == 8 * 8 * 8 * 2

    def test_float32_2d(self):
        assert get_slot_bytes((4, 4), "float32") == 4 * 4 * 4

    def test_uint8_1d(self):
        assert get_slot_bytes((16,), "uint8") == 16

    def test_numpy_dtype_passthrough(self):
        assert get_slot_bytes((2, 3), np.float32) == 2 * 3 * 4


class TestSlotInfoToView:
    """slot_info_to_view with real SharedMemory segments."""

    def test_single_slot_correctness(self):
        shape = (2, 4)
        dtype = np.dtype(np.uint16)
        shm, slot_bytes = _make_shm(2, shape, dtype)
        try:
            expected = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
            np.ndarray(shape, dtype=dtype, buffer=shm.buf, offset=0)[:] = expected

            view = slot_info_to_view(
                {"slot": 0, "slot_bytes": slot_bytes, "batch_shape": shape,
                 "dtype": dtype, "name": shm.name, "capacity": 2},
                shm,
            )
            np.testing.assert_array_equal(view, expected)
        finally:
            shm.close()
            shm.unlink()

    def test_offset_slot(self):
        shape = (2, 3)
        dtype = np.dtype(np.uint16)
        shm, slot_bytes = _make_shm(4, shape, dtype)
        try:
            expected = np.full(shape, 42, dtype=np.uint16)
            np.ndarray(
                shape, dtype=dtype, buffer=shm.buf, offset=2 * slot_bytes
            )[:] = expected

            view = slot_info_to_view(
                {"slot": 2, "slot_bytes": slot_bytes, "batch_shape": shape,
                 "dtype": dtype, "name": shm.name, "capacity": 4},
                shm,
            )
            np.testing.assert_array_equal(view, expected)
        finally:
            shm.close()
            shm.unlink()

    @pytest.mark.parametrize("np_dtype", [np.float32, np.uint8, np.uint16])
    def test_dtype_and_shape(self, np_dtype):
        shape = (3, 2)
        dtype = np.dtype(np_dtype)
        shm, slot_bytes = _make_shm(1, shape, dtype)
        try:
            view = slot_info_to_view(
                {"slot": 0, "slot_bytes": slot_bytes, "batch_shape": shape,
                 "dtype": dtype, "name": shm.name, "capacity": 1},
                shm,
            )
            assert view.dtype == dtype
            assert view.shape == shape
        finally:
            shm.close()
            shm.unlink()

    def test_view_is_writable(self):
        shape = (2, 2)
        dtype = np.dtype(np.uint16)
        shm, slot_bytes = _make_shm(1, shape, dtype)
        try:
            view = slot_info_to_view(
                {"slot": 0, "slot_bytes": slot_bytes, "batch_shape": shape,
                 "dtype": dtype, "name": shm.name, "capacity": 1},
                shm,
            )
            view[:] = 99
            np.testing.assert_array_equal(
                view, np.full(shape, 99, dtype=np.uint16)
            )
        finally:
            shm.close()
            shm.unlink()


# ===========================================================================
# Tier 2: HostMemoryBuffer Actor (requires Ray)
# ===========================================================================


class TestHostMemoryBuffer:
    """Integration tests for the HostMemoryBuffer Ray actor."""

    def test_get_free_returns_slot_info(self, ray_ctx, unique_suffix):
        name = f"hb_gf_{unique_suffix}"
        actor = _make_host_buffer(name)
        try:
            slot_info = ray.get(actor.get_free.remote())
            for key in (
                "slot",
                "name",
                "actor_name",
                "slot_bytes",
                "batch_shape",
                "dtype",
                "capacity",
            ):
                assert key in slot_info, f"missing key: {key}"
            assert slot_info["capacity"] == 4
            assert slot_info["batch_shape"] == (2, 4, 4)
            cfg = ray.get(actor.get_config.remote())
            assert slot_info["name"] == cfg["name"]
            assert slot_info["actor_name"] == name
        finally:
            _kill_safe(actor)

    def test_slot_reuse(self, ray_ctx, unique_suffix):
        name = f"hb_ru_{unique_suffix}"
        actor = _make_host_buffer(name, capacity=4)
        try:
            slots = [ray.get(actor.get_free.remote()) for _ in range(4)]
            assert {s["slot"] for s in slots} == {0, 1, 2, 3}
            for s in slots:
                ray.get(actor.put_free.remote(s["slot"]))
            slots2 = [ray.get(actor.get_free.remote()) for _ in range(4)]
            assert {s["slot"] for s in slots2} == {0, 1, 2, 3}
        finally:
            _kill_safe(actor)

    def test_try_get_free_returns_none_when_exhausted(self, ray_ctx, unique_suffix):
        name = f"hb_ex_{unique_suffix}"
        actor = _make_host_buffer(name, capacity=2)
        try:
            ray.get(actor.get_free.remote())
            ray.get(actor.get_free.remote())
            result = ray.get(actor.try_get_free.remote())
            assert result is None
        finally:
            _kill_safe(actor)

    def test_try_get_free_returns_slot_when_available(self, ray_ctx, unique_suffix):
        name = f"hb_tg_{unique_suffix}"
        actor = _make_host_buffer(name, capacity=2)
        try:
            result = ray.get(actor.try_get_free.remote())
            assert result is not None
            assert "slot" in result
        finally:
            _kill_safe(actor)

    def test_get_free_blocks_when_exhausted(self, ray_ctx, unique_suffix):
        """get_free blocks when all slots are in use, unblocks when one is freed."""
        name = f"hb_bl_{unique_suffix}"
        actor = _make_host_buffer(name, capacity=1)
        try:
            slot = ray.get(actor.get_free.remote())
            blocked_ref = actor.get_free.remote()
            ready, _ = ray.wait([blocked_ref], timeout=0.5)
            assert len(ready) == 0, "get_free should block when no slots available"

            ray.get(actor.put_free.remote(slot["slot"]))
            result = ray.get(blocked_ref, timeout=5.0)
            assert result["slot"] == slot["slot"]
        finally:
            _kill_safe(actor)

    def test_metrics_tracking(self, ray_ctx, unique_suffix):
        name = f"hb_mt_{unique_suffix}"
        actor = _make_host_buffer(name, capacity=4)
        try:
            slot = ray.get(actor.get_free.remote())
            ray.get(actor.put_free.remote(slot["slot"]))

            metrics = ray.get(actor.get_metrics.remote())
            assert len(metrics["get_free_wait_time_ms"]) == 1
            assert len(metrics["put_free_wait_time_ms"]) == 1
            assert metrics["occupied_slots"][-1] == 0
            assert sum(metrics["get_free_wait_time_ms"]) >= 0
            assert sum(metrics["put_free_wait_time_ms"]) >= 0
        finally:
            _kill_safe(actor)

    def test_clear_metrics(self, ray_ctx, unique_suffix):
        name = f"hb_cm_{unique_suffix}"
        actor = _make_host_buffer(name, capacity=4)
        try:
            ray.get(actor.get_free.remote())

            # First call returns accumulated metrics AND clears
            _ = ray.get(actor.get_metrics.remote())
            # Second call returns the cleared (empty) state
            metrics = ray.get(actor.get_metrics.remote())
            assert len(metrics["get_free_wait_time_ms"]) == 0
            assert len(metrics["put_free_wait_time_ms"]) == 0
            assert metrics["occupied_slots"] == []
            assert metrics["capacity"] == 4
        finally:
            _kill_safe(actor)

    def test_driver_attaches_shared_memory_using_actor_config(self, ray_ctx, unique_suffix):
        """Driver opens ``SharedMemory(name=cfg['name'])`` for the segment created in the actor (same attach path as BufferManager)."""
        name = f"hb_sm_{unique_suffix}"
        actor = _make_host_buffer(
            name, capacity=2, input_shape=(4,), batch_size=1, dtype="uint16"
        )
        local_shm = None
        try:
            cfg = ray.get(actor.get_config.remote())
            local_shm = shared_memory.SharedMemory(name=cfg["name"])
            arr = np.ndarray((1, 4), dtype=np.uint16, buffer=local_shm.buf)
            arr[:] = 42
            assert (arr == 42).all()
        finally:
            if local_shm is not None:
                local_shm.close()
            _kill_safe(actor)

    def test_get_config(self, ray_ctx, unique_suffix):
        name = f"hb_gc_{unique_suffix}"
        actor = _make_host_buffer(
            name, capacity=3, input_shape=(5,), batch_size=2, dtype="uint16"
        )
        try:
            cfg = ray.get(actor.get_config.remote())
            assert cfg["capacity"] == 3
            assert cfg["batch_shape"] == (2, 5)
            assert cfg["dtype"] == np.dtype(np.uint16)
            assert cfg["slot_bytes"] == int(np.prod((2, 5))) * 2
        finally:
            _kill_safe(actor)

    def test_multiple_get_put_cycles(self, ray_ctx, unique_suffix):
        """Repeatedly get and put the same slot to verify counter consistency."""
        name = f"hb_cyc_{unique_suffix}"
        actor = _make_host_buffer(name, capacity=1)
        n_cycles = 20
        try:
            for _ in range(n_cycles):
                slot = ray.get(actor.get_free.remote())
                ray.get(actor.put_free.remote(slot["slot"]))

            metrics = ray.get(actor.get_metrics.remote())
            assert len(metrics["get_free_wait_time_ms"]) == n_cycles
            assert len(metrics["put_free_wait_time_ms"]) == n_cycles
            assert metrics["occupied_slots"][-1] == 0
        finally:
            _kill_safe(actor)


# ===========================================================================
# Tier 3: BufferManager Integration (requires Ray, mocks CUDA)
# ===========================================================================


class TestBufferManagerSetGet:
    """set_buffer / get_buffer / remove_buffer lifecycle."""

    def test_set_buffer(self, ray_ctx, ray_node_id, unique_suffix):
        pool = f"sb_{unique_suffix}"
        bm = _make_buffer_manager(ray_node_id)
        actor = None
        try:
            actor, cfg = bm.set_buffer(
                pool_name=pool,
                batch_size=2,
                input_shape=(4, 4),
                dtype="uint16",
                buffer_type="host_memory",
                buffer_capacity=4,
                pin_numa_node=False,
            )
            assert pool in bm._buffer_actors
            assert pool in bm._buffer_cfgs
            assert pool in bm._buffer_shms
            assert bm._current_memory_usage_bytes > 0
            assert cfg["capacity"] == 4
            remote_cfg = ray.get(actor.get_config.remote())
            assert remote_cfg["capacity"] == 4
            assert remote_cfg["batch_shape"] == (2, 4, 4)
            assert "name" in remote_cfg
        finally:
            bm.shutdown()
            if actor:
                _kill_safe(actor)

    def test_set_buffer_duplicate_raises(self, ray_ctx, ray_node_id, unique_suffix):
        pool = f"dup_{unique_suffix}"
        bm = _make_buffer_manager(ray_node_id)
        actor = None
        try:
            actor, _ = bm.set_buffer(
                pool_name=pool, batch_size=2, input_shape=(4,),
                dtype="uint16", buffer_type="host_memory",
                buffer_capacity=2, pin_numa_node=False,
            )
            with pytest.raises(ValueError, match="already exists"):
                bm.set_buffer(
                    pool_name=pool, batch_size=2, input_shape=(4,),
                    dtype="uint16", buffer_type="host_memory",
                    buffer_capacity=2, pin_numa_node=False,
                )
        finally:
            bm.shutdown()
            if actor:
                _kill_safe(actor)

    def test_set_buffer_exceeds_budget(self, ray_ctx, ray_node_id, unique_suffix):
        pool = f"bud_{unique_suffix}"
        bm = _make_buffer_manager(ray_node_id, budget_gb=100 / (2 ** 30))
        try:
            with pytest.raises(ValueError, match="exceeds max memory"):
                bm.set_buffer(
                    pool_name=pool, batch_size=2, input_shape=(1024, 1024),
                    dtype="float32", buffer_type="host_memory",
                    buffer_capacity=4, pin_numa_node=False,
                )
        finally:
            bm.shutdown()

    def test_get_buffer_existing(self, ray_ctx, ray_node_id, unique_suffix):
        pool = f"gex_{unique_suffix}"
        bm = _make_buffer_manager(ray_node_id)
        actor = None
        try:
            actor, _ = bm.set_buffer(
                pool_name=pool, batch_size=2, input_shape=(4,),
                dtype="uint16", buffer_type="host_memory",
                buffer_capacity=2, pin_numa_node=False,
            )
            retrieved = bm.get_buffer(pool)
            assert retrieved is actor
        finally:
            bm.shutdown()
            if actor:
                _kill_safe(actor)

    def test_get_buffer_lazy_lookup(self, ray_ctx, ray_node_id, unique_suffix):
        """get_buffer discovers a pre-existing actor that was not created by this manager."""
        pool = f"lzy_{unique_suffix}"
        global_rank, numa_node = 0, 0
        namespace = f"buffers_node_{ray_node_id}"
        actor_name = get_buffer_name(pool, numa_node, global_rank)

        # Production set_buffers uses soft=False node affinity; soft=True here tolerates
        # placement on this single-node test cluster (multi-node is out of scope).
        actor = HostMemoryBuffer.options(
            name=actor_name,
            namespace=namespace,
            lifetime="detached",
            max_concurrency=10,
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=ray_node_id, soft=True,
            ),
        ).remote(
            name=actor_name, capacity=3, input_shape=(2,),
            batch_size=2, dtype="uint16",
            pin_numa_node=False, numa_node=numa_node,
        )
        ray.get(actor.get_config.remote())

        bm = _make_buffer_manager(
            ray_node_id, global_rank=global_rank, numa_node=numa_node,
        )
        try:
            retrieved = bm.get_buffer(pool)
            cfg = ray.get(retrieved.get_config.remote())
            assert cfg["capacity"] == 3
        finally:
            bm.shutdown()
            _kill_safe(actor)

    def test_remove_buffer(self, ray_ctx, ray_node_id, unique_suffix):
        """remove_buffer drops manager bookkeeping and closes local shm; it does not ray.kill the detached actor."""
        pool = f"rm_{unique_suffix}"
        bm = _make_buffer_manager(ray_node_id)
        actor = None
        try:
            actor, _ = bm.set_buffer(
                pool_name=pool, batch_size=2, input_shape=(4,),
                dtype="uint16", buffer_type="host_memory",
                buffer_capacity=2, pin_numa_node=False,
            )
            initial_bytes = bm._current_memory_usage_bytes
            bm.remove_buffer(pool)
            assert pool not in bm._buffer_actors
            assert pool not in bm._buffer_cfgs
            assert pool not in bm._buffer_shms
            assert bm._current_memory_usage_bytes < initial_bytes
        finally:
            bm.shutdown()
            if actor:
                _kill_safe(actor)

    def test_remove_nonexistent_raises(self, ray_ctx, ray_node_id):
        bm = _make_buffer_manager(ray_node_id)
        try:
            with pytest.raises(ValueError, match="does not exist"):
                bm.remove_buffer("nonexistent_pool")
        finally:
            bm.shutdown()


class TestBufferManagerSlotOps:
    """slot_info_to_view, free_slot, metrics."""

    def test_slot_info_to_view(self, ray_ctx, ray_node_id, unique_suffix):
        pool = f"vw_{unique_suffix}"
        bm = _make_buffer_manager(ray_node_id)
        actor = None
        try:
            actor, _ = bm.set_buffer(
                pool_name=pool, batch_size=2, input_shape=(4, 4),
                dtype="uint16", buffer_type="host_memory",
                buffer_capacity=4, pin_numa_node=False,
            )
            slot_info = ray.get(actor.get_free.remote())
            view = bm.slot_info_to_view(slot_info)
            assert isinstance(view, np.ndarray)
            assert view.shape == (2, 4, 4)
            assert view.dtype == np.uint16
        finally:
            bm.shutdown()
            if actor:
                _kill_safe(actor)

    def test_free_slot(self, ray_ctx, ray_node_id, unique_suffix):
        pool = f"fs_{unique_suffix}"
        bm = _make_buffer_manager(ray_node_id)
        actor = None
        try:
            actor, _ = bm.set_buffer(
                pool_name=pool, batch_size=2, input_shape=(4,),
                dtype="uint16", buffer_type="host_memory",
                buffer_capacity=1, pin_numa_node=False,
            )
            slot_info = ray.get(actor.get_free.remote())
            bm.free_slot(slot_info)
            slot_info2 = ray.get(actor.get_free.remote())
            assert slot_info2["slot"] == slot_info["slot"]
        finally:
            bm.shutdown()
            if actor:
                _kill_safe(actor)

    def test_get_metrics(self, ray_ctx, ray_node_id, unique_suffix):
        pool = f"gm_{unique_suffix}"
        bm = _make_buffer_manager(ray_node_id)
        actor = None
        try:
            actor, _ = bm.set_buffer(
                pool_name=pool, batch_size=2, input_shape=(4,),
                dtype="uint16", buffer_type="host_memory",
                buffer_capacity=2, pin_numa_node=False,
            )
            ray.get(actor.get_free.remote())
            metrics = bm.get_metrics()
            assert pool in metrics
            assert len(metrics[pool]["get_free_wait_time_ms"]) == 1
        finally:
            bm.shutdown()
            if actor:
                _kill_safe(actor)

    def test_clear_metrics(self, ray_ctx, ray_node_id, unique_suffix):
        pool = f"clm_{unique_suffix}"
        bm = _make_buffer_manager(ray_node_id)
        actor = None
        try:
            actor, _ = bm.set_buffer(
                pool_name=pool, batch_size=2, input_shape=(4,),
                dtype="uint16", buffer_type="host_memory",
                buffer_capacity=2, pin_numa_node=False,
            )
            ray.get(actor.get_free.remote())
            # First call returns accumulated metrics AND clears
            _ = ray.get(actor.get_metrics.remote())
            # Second call returns the cleared (empty) state
            direct = ray.get(actor.get_metrics.remote())
            assert len(direct["get_free_wait_time_ms"]) == 0
            assert len(direct["put_free_wait_time_ms"]) == 0
            assert direct["occupied_slots"] == []
            # bm.get_metrics() also clears; subsequent call returns empty
            _ = bm.get_metrics()
            metrics = bm.get_metrics()
            assert len(metrics[pool]["get_free_wait_time_ms"]) == 0
        finally:
            bm.shutdown()
            if actor:
                _kill_safe(actor)


class TestBufferManagerSerialization:
    """__getstate__ / __setstate__ round-trip and owner semantics."""

    def test_serialization_roundtrip(self, ray_ctx, ray_node_id, unique_suffix):
        pool = f"ser_{unique_suffix}"
        bm = _make_buffer_manager(ray_node_id)
        actor = None
        try:
            actor, _ = bm.set_buffer(
                pool_name=pool, batch_size=2, input_shape=(4,),
                dtype="uint16", buffer_type="host_memory",
                buffer_capacity=2, pin_numa_node=False,
            )
            data = pickle.dumps(bm)
            bm2 = pickle.loads(data)

            assert not bm2._is_owner
            assert pool in bm2._buffer_actors
            assert pool in bm2._buffer_shms

            slot = ray.get(bm2._buffer_actors[pool].get_free.remote())
            assert "slot" in slot

            bm2._cleanup_shms()
        finally:
            bm.shutdown()
            if actor:
                _kill_safe(actor)

    def test_nonowner_shutdown_preserves_actors(self, ray_ctx, ray_node_id, unique_suffix):
        pool = f"nown_{unique_suffix}"
        bm = _make_buffer_manager(ray_node_id)
        actor = None
        try:
            actor, _ = bm.set_buffer(
                pool_name=pool, batch_size=2, input_shape=(4,),
                dtype="uint16", buffer_type="host_memory",
                buffer_capacity=2, pin_numa_node=False,
            )
            bm2 = pickle.loads(pickle.dumps(bm))
            bm2.shutdown()

            cfg = ray.get(actor.get_config.remote())
            assert cfg["capacity"] == 2
        finally:
            bm.shutdown()
            if actor:
                _kill_safe(actor)

    def test_owner_shutdown_cleans_internal_state(self, ray_ctx, ray_node_id, unique_suffix):
        """Owner shutdown clears manager dicts; detached actors are still killed in finally via _kill_safe."""
        pool = f"osd_{unique_suffix}"
        bm = _make_buffer_manager(ray_node_id)
        actor = None
        try:
            actor, _ = bm.set_buffer(
                pool_name=pool, batch_size=2, input_shape=(4,),
                dtype="uint16", buffer_type="host_memory",
                buffer_capacity=2, pin_numa_node=False,
            )
            bm.shutdown()
            assert len(bm._buffer_actors) == 0
            assert len(bm._buffer_cfgs) == 0
            assert len(bm._buffer_shms) == 0
        finally:
            if actor:
                _kill_safe(actor)

    def test_shutdown_idempotent(self, ray_ctx, ray_node_id, unique_suffix):
        pool = f"idm_{unique_suffix}"
        bm = _make_buffer_manager(ray_node_id)
        actor = None
        try:
            actor, _ = bm.set_buffer(
                pool_name=pool, batch_size=2, input_shape=(4,),
                dtype="uint16", buffer_type="host_memory",
                buffer_capacity=2, pin_numa_node=False,
            )
            bm.shutdown()
            bm.shutdown()  # must not raise
        finally:
            if actor:
                _kill_safe(actor)


# ===========================================================================
# init_output_memory_pools
# ===========================================================================


class TestInitOutputMemoryPools:

    def test_save_only(self, ray_ctx, ray_node_id, unique_suffix):
        bm = _make_buffer_manager(ray_node_id, global_rank=hash(unique_suffix) % 10000)
        actors = []
        try:
            meta = {
                "tensor_info": {"seg": {"shape": (4,), "dtype": "uint16"}},
                "save_tensors": ["seg"],
                "visualize_tensors": [],
                "buffer_tensors": ["seg"],
            }
            init_output_memory_pools(
                buffer_manager=bm, output_metadata=meta, batch_size=2,
                save=True, viz=False, save_buffer_capacity=2,
                pin_numa_node=False,
            )
            assert "seg_save" in bm._buffer_actors
            actors.append(bm._buffer_actors["seg_save"])
        finally:
            bm.shutdown()
            for a in actors:
                _kill_safe(a)

    def test_viz_only(self, ray_ctx, ray_node_id, unique_suffix):
        rank = (hash(unique_suffix) % 10000) + 10000
        bm = _make_buffer_manager(ray_node_id, global_rank=rank)
        actors = []
        try:
            meta = {
                "tensor_info": {"conf": {"shape": (4,), "dtype": "float32"}},
                "save_tensors": [],
                "visualize_tensors": ["conf"],
                "buffer_tensors": ["conf"],
            }
            init_output_memory_pools(
                buffer_manager=bm, output_metadata=meta, batch_size=2,
                save=False, viz=True, viz_buffer_capacity=2,
                pin_numa_node=False,
            )
            assert "conf_viz" in bm._buffer_actors
            actors.append(bm._buffer_actors["conf_viz"])
        finally:
            bm.shutdown()
            for a in actors:
                _kill_safe(a)

    def test_both_save_and_viz(self, ray_ctx, ray_node_id, unique_suffix):
        rank = (hash(unique_suffix) % 10000) + 20000
        bm = _make_buffer_manager(ray_node_id, global_rank=rank)
        actors = []
        try:
            meta = {
                "tensor_info": {"out": {"shape": (4,), "dtype": "uint16"}},
                "save_tensors": ["out"],
                "visualize_tensors": ["out"],
                "buffer_tensors": ["out"],
            }
            init_output_memory_pools(
                buffer_manager=bm, output_metadata=meta, batch_size=2,
                save=True, viz=True,
                save_buffer_capacity=2, viz_buffer_capacity=2,
                pin_numa_node=False,
            )
            assert "out_save" in bm._buffer_actors
            assert "out_viz" in bm._buffer_actors
            actors.extend([
                bm._buffer_actors["out_save"],
                bm._buffer_actors["out_viz"],
            ])
        finally:
            bm.shutdown()
            for a in actors:
                _kill_safe(a)

    def test_empty_buffer_tensors_creates_no_pools(self, ray_ctx, ray_node_id, unique_suffix):
        """Small tensors may be listed in save_tensors but omitted from buffer_tensors."""
        rank = (hash(unique_suffix) % 10000) + 21000
        bm = _make_buffer_manager(ray_node_id, global_rank=rank)
        try:
            meta = {
                "tensor_info": {"tiny": {"shape": (2,), "dtype": "float32"}},
                "save_tensors": ["tiny"],
                "visualize_tensors": [],
                "buffer_tensors": [],
            }
            init_output_memory_pools(
                buffer_manager=bm, output_metadata=meta, batch_size=2,
                save=True, viz=False, save_buffer_capacity=2,
                pin_numa_node=False,
            )
            assert "tiny_save" not in bm._buffer_actors
        finally:
            bm.shutdown()

    def test_missing_tensor_info_raises(self, ray_ctx, ray_node_id, unique_suffix):
        rank = (hash(unique_suffix) % 10000) + 30000
        bm = _make_buffer_manager(ray_node_id, global_rank=rank)
        try:
            meta = {
                "tensor_info": {},
                "save_tensors": ["nonexistent"],
                "visualize_tensors": [],
                "buffer_tensors": ["nonexistent"],
            }
            with pytest.raises(ValueError, match="Tensor info for nonexistent"):
                init_output_memory_pools(
                    buffer_manager=bm, output_metadata=meta, batch_size=2,
                    save=True, viz=False, save_buffer_capacity=2,
                    pin_numa_node=False,
                )
        finally:
            bm.shutdown()

    def test_neither_save_nor_viz_raises(self, ray_ctx, ray_node_id):
        bm = _make_buffer_manager(ray_node_id)
        try:
            with pytest.raises(ValueError, match="at least one"):
                init_output_memory_pools(
                    buffer_manager=bm,
                    output_metadata={
                        "tensor_info": {},
                        "save_tensors": [],
                        "visualize_tensors": [],
                        "buffer_tensors": [],
                    },
                    batch_size=2,
                    save=False, viz=False,
                )
        finally:
            bm.shutdown()

    def test_missing_save_capacity_raises(self, ray_ctx, ray_node_id):
        bm = _make_buffer_manager(ray_node_id)
        try:
            with pytest.raises(ValueError, match="save_buffer_capacity"):
                init_output_memory_pools(
                    buffer_manager=bm,
                    output_metadata={
                        "tensor_info": {},
                        "save_tensors": [],
                        "visualize_tensors": [],
                        "buffer_tensors": [],
                    },
                    batch_size=2,
                    save=True, viz=False, save_buffer_capacity=None,
                )
        finally:
            bm.shutdown()

    def test_missing_viz_capacity_raises(self, ray_ctx, ray_node_id):
        bm = _make_buffer_manager(ray_node_id)
        try:
            with pytest.raises(ValueError, match="viz_buffer_capacity"):
                init_output_memory_pools(
                    buffer_manager=bm,
                    output_metadata={
                        "tensor_info": {},
                        "save_tensors": [],
                        "visualize_tensors": [],
                        "buffer_tensors": [],
                    },
                    batch_size=2,
                    save=False, viz=True, viz_buffer_capacity=None,
                )
        finally:
            bm.shutdown()


# ===========================================================================
# set_buffers (module-level function)
# ===========================================================================


class TestSetBuffers:

    def test_creates_actor(self, ray_ctx, ray_node_id, unique_suffix):
        pool = f"sbf_{unique_suffix}"
        actor = None
        try:
            actor, cfg = set_buffers(
                local_rank=0, global_rank=0, numa_node=0,
                dtype=np.uint16, batch_size=2,
                input_shape=(4,), buffer_type="host_memory",
                buffer_capacity=3, pin_numa_node=False,
                node_id=ray_node_id, pool_name=pool,
            )
            assert cfg["capacity"] == 3
            assert cfg["batch_shape"] == (2, 4)
        finally:
            if actor:
                _kill_safe(actor)

    def test_idempotent_reuse(self, ray_ctx, ray_node_id, unique_suffix):
        pool = f"sbfi_{unique_suffix}"
        actor = None
        try:
            actor, cfg1 = set_buffers(
                local_rank=0, global_rank=0, numa_node=0,
                dtype=np.uint16, batch_size=2,
                input_shape=(4,), buffer_type="host_memory",
                buffer_capacity=3, pin_numa_node=False,
                node_id=ray_node_id, pool_name=pool,
            )
            _, cfg2 = set_buffers(
                local_rank=0, global_rank=0, numa_node=0,
                dtype=np.uint16, batch_size=2,
                input_shape=(4,), buffer_type="host_memory",
                buffer_capacity=3, pin_numa_node=False,
                node_id=ray_node_id, pool_name=pool,
            )
            assert cfg1["name"] == cfg2["name"]
        finally:
            if actor:
                _kill_safe(actor)

    def test_config_mismatch_raises(self, ray_ctx, ray_node_id, unique_suffix):
        pool = f"sbfm_{unique_suffix}"
        actor = None
        try:
            actor, _ = set_buffers(
                local_rank=0, global_rank=0, numa_node=0,
                dtype=np.uint16, batch_size=2,
                input_shape=(4,), buffer_type="host_memory",
                buffer_capacity=3, pin_numa_node=False,
                node_id=ray_node_id, pool_name=pool,
            )
            with pytest.raises(ValueError, match="config does not match"):
                set_buffers(
                    local_rank=0, global_rank=0, numa_node=0,
                    dtype=np.uint16, batch_size=2,
                    input_shape=(4,), buffer_type="host_memory",
                    buffer_capacity=99,
                    pin_numa_node=False,
                    node_id=ray_node_id, pool_name=pool,
                )
        finally:
            if actor:
                _kill_safe(actor)

    def test_unsupported_type_raises(self, ray_ctx, ray_node_id, unique_suffix):
        with pytest.raises(ValueError, match="Unsupported buffer type"):
            set_buffers(
                local_rank=0, global_rank=0, numa_node=0,
                dtype=np.uint16, batch_size=2,
                input_shape=(4,), buffer_type="gpu_memory",
                buffer_capacity=3, pin_numa_node=False,
                node_id=ray_node_id, pool_name=f"sbfu_{unique_suffix}",
            )


# ===========================================================================
# Leak Detection
# ===========================================================================


class TestLeakDetection:

    def test_no_shm_leak_after_full_lifecycle(self, ray_ctx, ray_node_id, unique_suffix):
        """SharedMemory segment should be released after actor termination."""
        pool = f"lk_{unique_suffix}"
        bm = _make_buffer_manager(ray_node_id)
        actor = None
        shm_name = None
        try:
            actor, cfg = bm.set_buffer(
                pool_name=pool, batch_size=2, input_shape=(4,),
                dtype="uint16", buffer_type="host_memory",
                buffer_capacity=2, pin_numa_node=False,
            )
            shm_name = cfg["name"]

            slot = ray.get(actor.get_free.remote())
            ray.get(actor.put_free.remote(slot["slot"]))

            bm.shutdown()
            assert len(bm._buffer_actors) == 0
            assert len(bm._buffer_cfgs) == 0
            assert len(bm._buffer_shms) == 0
            # Actor is still running; segment still owned by worker until teardown.
            assert ray.get(actor.get_config.remote())["name"] == shm_name
            # Simulate graceful exit by calling __ray_terminate__ and waiting for actor to exit.
            actor.__ray_terminate__.remote()

            deadline = time.monotonic() + 5.0
            cleaned = False
            while time.monotonic() < deadline:
                try:
                    probe = shared_memory.SharedMemory(name=shm_name)
                    probe.close()
                    time.sleep(0.3)
                except FileNotFoundError:
                    cleaned = True
                    break

            if not cleaned:
                try:
                    leftover = shared_memory.SharedMemory(name=shm_name)
                except FileNotFoundError:
                    return
                leftover.close()
                leftover.unlink()
                pytest.fail(
                    "SharedMemory was still present after __ray_terminate__ + 5s grace; "
                    "expected HostMemoryBuffer atexit/_cleanup to unlink it. "
                    "Manual unlink was applied so the OS segment does not leak to later tests."
                )
        finally:
            if actor:
                _kill_safe(actor)

    def test_no_actor_leak_after_kill(self, ray_ctx, ray_node_id, unique_suffix):
        """Detached actors are not retrievable after explicit kill."""
        pool = f"alk_{unique_suffix}"
        bm = _make_buffer_manager(ray_node_id)
        actor = None
        try:
            actor, _ = bm.set_buffer(
                pool_name=pool, batch_size=2, input_shape=(4,),
                dtype="uint16", buffer_type="host_memory",
                buffer_capacity=2, pin_numa_node=False,
            )
            actor_name = get_buffer_name(pool, 0, 0)
            namespace = f"buffers_node_{ray_node_id}"

            bm.shutdown()
            ray.kill(actor)
            actor = None
            time.sleep(0.5)

            # ray.get_actor raises ValueError when the named actor is not registered.
            with pytest.raises(ValueError):
                ray.get_actor(actor_name, namespace=namespace)
        finally:
            if actor:
                _kill_safe(actor)
