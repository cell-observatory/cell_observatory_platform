"""
Output layout contract for packed inference slots.

Defines OutputLayoutEntry / OutputLayout schema, deterministic layout computation,
validation, and JSON-safe serialization for run-manifest persistence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np


# Supported dtypes for layout computation (NumPy-compatible names)
_SUPPORTED_DTYPES = frozenset({"float32", "float16", "bfloat16", "int16", "int32", "uint8"})
_SUPPORTED_ORDERS = frozenset({"C", "F"})


@dataclass(frozen=True)
class OutputLayoutEntry:
    """Single output entry in a packed slot layout."""

    output_name: str
    output_type: str
    dtype: str
    shape: tuple[int, ...]
    order: str
    offset_bytes: int
    nbytes: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict for manifest persistence."""
        return {
            "output_name": self.output_name,
            "output_type": self.output_type,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "order": self.order,
            "offset_bytes": self.offset_bytes,
            "nbytes": self.nbytes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OutputLayoutEntry:
        """Reconstruct from JSON-safe dict."""
        return cls(
            output_name=str(d["output_name"]),
            output_type=str(d["output_type"]),
            dtype=str(d["dtype"]),
            shape=tuple(int(x) for x in d["shape"]),
            order=str(d["order"]),
            offset_bytes=int(d["offset_bytes"]),
            nbytes=int(d["nbytes"]),
        )


@dataclass
class OutputLayout:
    """Layout manifest for a packed uint8 slot buffer."""

    entries: list[OutputLayoutEntry] = field(default_factory=list)
    slot_bytes_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict for manifest persistence."""
        return {
            "slot_bytes_total": self.slot_bytes_total,
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OutputLayout:
        """Reconstruct from JSON-safe dict."""
        entries = [OutputLayoutEntry.from_dict(e) for e in d.get("entries", [])]
        slot_bytes_total = int(d.get("slot_bytes_total", 0))
        return cls(entries=entries, slot_bytes_total=slot_bytes_total)


def _dtype_itemsize(dtype_str: str) -> int:
    """Return itemsize in bytes for a dtype string."""
    return int(np.dtype(dtype_str).itemsize)


def validate_layout(layout: OutputLayout) -> None:
    """
    Validate layout for deterministic pack/unpack safety.
    Raises ValueError if invalid.
    """
    if not layout.entries:
        raise ValueError("Layout must have at least one entry")

    for i, entry in enumerate(layout.entries):
        if not entry.output_name:
            raise ValueError(f"Entry {i}: output_name must be non-empty")
        if not entry.output_type:
            raise ValueError(f"Entry {i}: output_type must be non-empty")
        if entry.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"Entry {i} ({entry.output_name}): dtype {entry.dtype!r} not supported; "
                f"allowed: {sorted(_SUPPORTED_DTYPES)}"
            )
        if entry.order not in _SUPPORTED_ORDERS:
            raise ValueError(
                f"Entry {i} ({entry.output_name}): order {entry.order!r} not supported; "
                f"allowed: {sorted(_SUPPORTED_ORDERS)}"
            )
        if not entry.shape:
            raise ValueError(f"Entry {i} ({entry.output_name}): shape must be non-empty")
        if any(s <= 0 for s in entry.shape):
            raise ValueError(
                f"Entry {i} ({entry.output_name}): all shape dims must be positive, got {entry.shape}"
            )
        if entry.nbytes <= 0:
            raise ValueError(
                f"Entry {i} ({entry.output_name}): nbytes must be positive, got {entry.nbytes}"
            )
        expected_nbytes = math.prod(entry.shape) * _dtype_itemsize(entry.dtype)
        if entry.nbytes != expected_nbytes:
            raise ValueError(
                f"Entry {i} ({entry.output_name}): nbytes {entry.nbytes} does not match "
                f"prod(shape)*itemsize = {expected_nbytes}"
            )
        if entry.offset_bytes < 0:
            raise ValueError(
                f"Entry {i} ({entry.output_name}): offset_bytes must be non-negative, "
                f"got {entry.offset_bytes}"
            )

    # No overlaps: ranges [offset, offset+nbytes) must be disjoint
    ranges: list[tuple[int, int]] = [
        (e.offset_bytes, e.offset_bytes + e.nbytes) for e in layout.entries
    ]
    ranges.sort(key=lambda r: r[0])
    for j in range(1, len(ranges)):
        if ranges[j][0] < ranges[j - 1][1]:
            raise ValueError(
                f"Overlapping layout ranges: {ranges[j - 1]} and {ranges[j]}"
            )

    # Sum check
    total = sum(e.nbytes for e in layout.entries)
    if total != layout.slot_bytes_total:
        raise ValueError(
            f"sum(entry.nbytes)={total} != slot_bytes_total={layout.slot_bytes_total}"
        )


def compute_layout(
    outputs_metadata: dict[str, dict[str, Any]],
    output_type_configs: dict[str, dict[str, Any]],
) -> OutputLayout:
    """
    Compute deterministic layout from outputs_metadata and output_type configs.

    Args:
        outputs_metadata: {output_name: {output_type, shape, path?, ...}}
        output_type_configs: {output_type_name: {dtype, order, save?, viz?}}

    Returns:
        Validated OutputLayout with contiguous offsets.
    """
    entries: list[OutputLayoutEntry] = []
    offset = 0

    # Iterate in deterministic order (insertion order of outputs_metadata)
    for output_name, meta in outputs_metadata.items():
        output_type = meta.get("output_type")
        if output_type is None:
            raise ValueError(
                f"outputs_metadata[{output_name!r}]: missing 'output_type'"
            )
        output_type = str(output_type)

        shape_raw = meta.get("shape")
        if shape_raw is None:
            raise ValueError(
                f"outputs_metadata[{output_name!r}]: missing 'shape'"
            )
        shape = tuple(int(x) for x in shape_raw)

        type_cfg = output_type_configs.get(output_type)
        if type_cfg is None:
            raise ValueError(
                f"outputs_metadata[{output_name!r}]: output_type {output_type!r} "
                "not found in output_type_configs"
            )

        dtype = type_cfg.get("dtype", "float32")
        order = type_cfg.get("order", "C")
        dtype = str(dtype)
        order = str(order)

        nbytes = math.prod(shape) * _dtype_itemsize(dtype)
        entry = OutputLayoutEntry(
            output_name=output_name,
            output_type=output_type,
            dtype=dtype,
            shape=shape,
            order=order,
            offset_bytes=offset,
            nbytes=nbytes,
        )
        entries.append(entry)
        offset += nbytes

    slot_bytes_total = offset
    layout = OutputLayout(entries=entries, slot_bytes_total=slot_bytes_total)
    validate_layout(layout)
    return layout


def layout_to_manifest_dict(layout: Union[OutputLayout, dict]) -> dict[str, Any]:
    """
    Produce canonical JSON payload for manifest persistence.
    Accepts OutputLayout or plain dict (from OmegaConf/Hydra).
    """
    if isinstance(layout, OutputLayout):
        return layout.to_dict()
    # Plain dict from config
    if isinstance(layout, dict):
        entries = layout.get("entries", [])
        slot_bytes_total = layout.get("slot_bytes_total", 0)
        return {
            "slot_bytes_total": int(slot_bytes_total),
            "entries": [
                {
                    "output_name": str(e.get("output_name", "")),
                    "output_type": str(e.get("output_type", "")),
                    "dtype": str(e.get("dtype", "float32")),
                    "shape": [int(x) for x in e.get("shape", [])],
                    "order": str(e.get("order", "C")),
                    "offset_bytes": int(e.get("offset_bytes", 0)),
                    "nbytes": int(e.get("nbytes", 0)),
                }
                for e in entries
            ],
        }
    raise TypeError(f"layout must be OutputLayout or dict, got {type(layout)}")
