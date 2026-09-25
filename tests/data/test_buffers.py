"""Device-buffer slot bookkeeping (CPU: the buffer object is built without its
CUDA tensors)."""
import queue

import pytest

from cell_observatory_platform.data.datasets.buffers import DeviceMemoryBuffer


def _bare(capacity=2):
    b = object.__new__(DeviceMemoryBuffer)
    b.name, b.capacity = "device_buffer_rank_0", capacity
    b._free = queue.SimpleQueue()
    for i in range(capacity):
        b._free.put(i)
    return b


def test_slots_cycle_through_the_free_queue():
    b = _bare(2)
    assert (b.get_free(), b.get_free()) == (0, 1)
    b.put_free(1)
    assert b.get_free() == 1


def test_exhausted_buffer_fails_loudly_instead_of_hanging():
    b = _bare(1)
    b.slot_wait_timeout_s = 0.05
    assert b.get_free() == 0
    with pytest.raises(RuntimeError, match="no free device-buffer slot"):
        b.get_free()
