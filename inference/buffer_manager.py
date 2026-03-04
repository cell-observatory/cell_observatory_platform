"""
Per-rank BufferManager: owns shared-memory pools and mediates allocation/freeing.

Phase 0B implements save_output pool (wraps set_output_buffers). Viz output is stubbed.
preproc_input and postproc_input are deferred to data-loading refactor.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import ray

from cell_observatory_platform.data.datasets.buffers import set_output_buffers

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SlotHandle:
    """Handle for an allocated slot in a BufferManager pool."""

    pool_class: str
    slot_idx: int
    shm_name: str
    slot_bytes: int
    batch_size: int
    batch_shape: tuple


class BufferManager:
    """
    Per-rank memory manager. Owns save_output pool (wraps HostMemoryBuffer).
    preproc_input, postproc_input: not implemented (deferred).
    viz_output: stub (alloc returns None when empty; non-blocking).
    """

    def __init__(
        self,
        *,
        local_rank: int,
        global_rank: int,
        node_id: int,
        numa_node: int,
        batch_size: int,
        slot_bytes_total: int,
        buffer_capacity: int = 8,
        rank_memory_budget_gb: float = 64.0,
        memory_fractions: Optional[Dict[str, float]] = None,
    ):
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.node_id = node_id
        self.numa_node = numa_node
        self.batch_size = batch_size
        self.slot_bytes_total = slot_bytes_total
        self.buffer_capacity = buffer_capacity
        self.rank_memory_budget_gb = rank_memory_budget_gb
        self._memory_fractions = memory_fractions or {
            "pre": 0.1,
            "post": 0.1,
            "save": 0.7,
            "viz": 0.1,
        }

        # Create save_output pool via set_output_buffers
        self._output_buffer_actor, self._output_buffer_cfg = set_output_buffers(
            local_rank=local_rank,
            global_rank=global_rank,
            numa_node=numa_node,
            batch_size=batch_size,
            output_shape=(slot_bytes_total,),
            dtype="uint8",
            buffer_capacity=buffer_capacity,
            pin_to_numa_node=True,
            node_id=node_id,
        )

        # Metrics
        self._save_in_use_high_water: int = 0
        self._save_in_use_current: int = 0
        self._save_alloc_count: int = 0
        self._save_blocked_time_s: float = 0.0
        self._viz_drops: int = 0

    def alloc(self, pool_class: str, block: bool = True) -> Optional[SlotHandle]:
        """
        Allocate a slot from the given pool.

        - save_output: blocking (block=True). Returns SlotHandle.
        - viz_output: non-blocking (block=False). Returns None when pool empty.
        - preproc_input, postproc_input: not implemented, raises.
        """
        if pool_class == "save_output":
            t0 = time.perf_counter()
            result = ray.get(self._output_buffer_actor.get_free.remote())
            blocked = time.perf_counter() - t0
            self._save_blocked_time_s += blocked
            self._save_alloc_count += 1
            self._save_in_use_current += 1
            if self._save_in_use_current > self._save_in_use_high_water:
                self._save_in_use_high_water = self._save_in_use_current

            # result is dict: slot, name, slot_bytes, batch_shape, dtype, capacity
            return SlotHandle(
                pool_class="save_output",
                slot_idx=result["slot"],
                shm_name=result["name"],
                slot_bytes=result["slot_bytes"],
                batch_size=self.batch_size,
                batch_shape=tuple(result["batch_shape"]),
            )
        elif pool_class == "viz_output":
            if not block:
                # Stub: no viz pool yet, always return None
                self._viz_drops += 1
                return None
            raise NotImplementedError("viz_output blocking alloc not supported")
        elif pool_class in ("preproc_input", "postproc_input"):
            raise NotImplementedError(
                f"{pool_class} pool not implemented; deferred to data-loading refactor"
            )
        else:
            raise ValueError(f"Unknown pool_class: {pool_class}")

    def free(self, pool_class: str, handle: SlotHandle) -> None:
        """Return a slot to the pool."""
        if pool_class != handle.pool_class:
            raise ValueError(
                f"handle pool_class {handle.pool_class} != free pool_class {pool_class}"
            )
        if pool_class == "save_output":
            ray.get(self._output_buffer_actor.put_free.remote(handle.slot_idx))
            self._save_in_use_current = max(0, self._save_in_use_current - 1)
        else:
            raise ValueError(f"free not implemented for pool_class: {pool_class}")

    def get_metrics(self) -> Dict[str, Any]:
        """Return metrics for shutdown logging."""
        return {
            "save_output": {
                "alloc_count": self._save_alloc_count,
                "high_water_slots": self._save_in_use_high_water,
                "blocked_time_s": self._save_blocked_time_s,
            },
            "viz_output": {
                "drops": self._viz_drops,
            },
        }

    def log_metrics_at_shutdown(self) -> None:
        """Log BufferManager metrics at shutdown."""
        metrics = self.get_metrics()
        logger.info(
            "BufferManager shutdown metrics: %s",
            metrics,
        )

    @property
    def output_buffer_cfg(self) -> Dict[str, Any]:
        """Expose output buffer config for callers that need shm_name, slot_bytes."""
        return dict(self._output_buffer_cfg)
