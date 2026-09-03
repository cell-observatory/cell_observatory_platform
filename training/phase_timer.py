"""Per-step phase timers on CUDA events, logged next to the perf metrics.

Opt-in: the trainer calls ``PhaseTimer.enable()`` when ``trainer_loop.with_perf_metrics`` is on;
otherwise every ``phase``/``timed`` is a no-op. Phases nest freely (each is its own event pair) and
repeated phases within a step are summed. ``collect()`` must run after a device sync (the perf
metrics processor already does one) and returns ``{"<name>_ms": float, "<count>": float}``.
"""

from __future__ import annotations

import functools
from collections import defaultdict
from contextlib import contextmanager

import torch


class PhaseTimer:
    _enabled: bool = False
    _events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
    _counts: dict[str, float] = defaultdict(float)

    @classmethod
    def enable(cls, on: bool = True) -> None:
        cls._enabled = bool(on) and torch.cuda.is_available()

    @classmethod
    @contextmanager
    def phase(cls, name: str):
        if not cls._enabled:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            cls._events[name].append((start, end))

    @classmethod
    def timed(cls, name: str):
        def deco(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                with cls.phase(name):
                    return fn(*args, **kwargs)
            return wrapper
        return deco

    @classmethod
    def count(cls, name: str, value: float) -> None:
        if cls._enabled:
            cls._counts[name] += float(value)

    @classmethod
    def collect(cls) -> dict[str, float]:
        """Sum elapsed ms per phase for the step; caller must have synced the device."""
        out: dict[str, float] = {}
        for name, pairs in cls._events.items():
            out[f"{name}_ms"] = sum(s.elapsed_time(e) for s, e in pairs)
        for name, v in cls._counts.items():
            out[name] = v
        cls._events.clear()
        cls._counts.clear()
        return out
