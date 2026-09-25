"""Zero (and optionally permute) a random subset of signal channels per sample."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch

from cell_observatory_platform.data.datasets.utils import _as_list, _normalize_channel_token


class ChannelDropout:
    """Channel dropout for the SAM2 preprocessor chain (after ``Normalize``).

    Runs on the signal-only tensor ``(B, [T,] Z, Y, X, C)``. Each droppable
    channel is zeroed with probability ``p`` per sample; at least one droppable
    channel always survives. With ``shuffle`` the droppable channels are also
    permuted (per sample) before the drop. ``C`` never changes, so
    ``channel_mapping`` and the patch embed are untouched.

    ``always_keep`` names channels that are never zeroed or moved, either by
    emitted position (ints) or by token (strings matched against the
    localization OR fluorophore in ``metainfo["channel_tokens"]``, the
    post-selection ``[localization, fluorophore]`` list the loader emits).

    When ``metainfo["channel_ids"]`` is present (channel-adaptive embed, set
    by the SAM2 preprocessor before its transforms run) a shuffle permutes
    those rows identically, so the channel embedding follows the data.
    """

    reads_raw_counts = False

    def __init__(
        self,
        p: float = 0.25,
        always_keep: Sequence[Union[int, str]] = ("membrane",),
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if not 0.0 <= float(p) <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}")
        self.p = float(p)
        self.keep_positions = {int(c) for c in always_keep if not isinstance(c, str)}
        self.keep_tokens = {_normalize_channel_token(c) for c in always_keep if isinstance(c, str)}
        self.shuffle = bool(shuffle)
        self.rng = torch.Generator()
        self.rng.manual_seed(int(seed) if seed is not None else 0)

    # ------------------------------------------------------------------ #
    def _keep_for_sample(self, meta: Dict[str, Any], b: int, C: int) -> set[int]:
        keep = {c for c in self.keep_positions if 0 <= c < C}
        if not self.keep_tokens:
            return keep
        tokens = meta.get("channel_tokens")
        if tokens is None:
            raise ValueError(
                f"ChannelDropout always_keep names tokens {sorted(self.keep_tokens)} but "
                "metainfo has no 'channel_tokens' (the loader emits it next to channel_mapping)"
            )
        # one entry per sample (JSON string or list), as the collator carries it
        row = _as_list(tokens[b]) or []
        found = False
        for pos, tok in enumerate(row[:C]):
            if tok is None:
                continue
            pair = {_normalize_channel_token(t) for t in _as_list(tok) or [] if t is not None}
            if pair & self.keep_tokens:
                keep.add(pos)
                found = True
        if not found:
            raise ValueError(
                f"ChannelDropout always_keep tokens {sorted(self.keep_tokens)} match no signal "
                f"channel in sample {b}: channel_tokens={row[:C]!r}"
            )
        return keep

    def _coin(self) -> float:
        return float(torch.rand((), generator=self.rng))

    # ------------------------------------------------------------------ #
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        x = data["data_tensor"]
        meta = data.get("metainfo") or {}
        B, C = int(x.shape[0]), int(x.shape[-1])
        ids = meta.get("channel_ids")

        for b in range(B):
            keep = self._keep_for_sample(meta, b, C)
            droppable = [c for c in range(C) if c not in keep]
            if not droppable:
                continue

            if self.shuffle and len(droppable) > 1:
                perm = torch.randperm(len(droppable), generator=self.rng).tolist()
                src = [droppable[i] for i in perm]
                x[b, ..., droppable] = x[b, ..., src].clone()
                if ids is not None:
                    ids[b, droppable] = ids[b, src].clone()

            drop = [c for c in droppable if self._coin() < self.p]
            if drop and len(drop) == len(droppable):
                drop = drop[:-1]                      # never zero every droppable channel
            if drop:
                x[b, ..., drop] = 0

        data["data_tensor"] = x
        return data
