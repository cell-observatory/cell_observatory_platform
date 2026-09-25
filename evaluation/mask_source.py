"""Mask sources for instance-segmentation evaluation.

Abstracts *where* per-instance binary masks come from so the evaluator does not
branch on model internals:

* :class:`QueryMaskSource` - query-embedding detectors (e.g: MaskDINO): lazily
  materialize and binarize mask logits via the chunked
  :class:`~cell_observatory_platform.models.meta_arch.maskdino.MaskMaterializer`.
* :class:`DirectMaskSource` - models that already emit hard, per-instance binary
  masks (e.g: SAM2 AMG): slice/chunk the precomputed bool tensor.

:func:`build_mask_source` dispatches on the ``mask_source`` key the model
DECLARES in its ``predict_for_eval`` sample (``"direct"`` or ``"query"``), never by
sniffing which keys happen to be present. The rank/dtype checks are retained, but
as assertions on the declared contract rather than as the routing mechanism: a
model producing *soft* semantic probabilities (e.g. Mask2Former's float
``pred_masks``) must NOT be reinterpreted as hard instance masks -- ``.bool()`` on
floats would yield garbage IoU. Those belong to the semantic evaluator, so a
floating ``pred_masks`` still raises here.

Both sources yield ``(k, D, H, W)`` bool tensors on a caller-provided ``device``,
chunked to bound peak memory at ``chunk_size * D*H*W`` regardless of the model.
"""

from abc import ABC, abstractmethod
from typing import Any, Iterator, Literal, Mapping, Optional, Sequence

import torch

from cell_observatory_platform.models.layers.maskmaterializer import MaskMaterializer


# The mask source a model declares in its predict_for_eval sample under "mask_source".
MaskSourceKind = Literal["direct", "query"]


class MaskSource(ABC):
    """Yields chunks of per-instance binary masks for an evaluator image."""

    @abstractmethod
    def binary_mask_chunks(
        self,
        query_indices: torch.Tensor,
        chunk_size: int,
        device: torch.device,
    ) -> Iterator[torch.Tensor]:
        """Yield ``(k, D, H, W)`` bool tensors on ``device`` (``k <= chunk_size``).

        ``query_indices`` selects which instances/queries to materialize (a
        subset of all proposals, e.g. one class slice).
        """
        raise NotImplementedError


class QueryMaskSource(MaskSource):
    """Query-embedding source (e.g: MaskDINO): einsum + trilinear upsample + threshold.

    Wraps :class:`MaskMaterializer` so the evaluator never owns trilinear upsample
    / binarization details. Logits are binarized at ``> 0`` (i.e. sigmoid ``> 0.5``),
    matching the dense ``MaskMIoUMetric``/``MaskMAPMetric`` path.
    """

    def __init__(
        self,
        mask_embeddings: torch.Tensor,
        pixel_decoder_output: torch.Tensor,
        target_size: Sequence[int],
        upsample_dtype: torch.dtype = torch.float32,
    ) -> None:
        self._materializer = MaskMaterializer(
            mask_embeddings=mask_embeddings,
            pixel_decoder_output=pixel_decoder_output,
            target_size=target_size,
            upsample_dtype=upsample_dtype,
        )

    def binary_mask_chunks(
        self,
        query_indices: torch.Tensor,
        chunk_size: int,
        device: torch.device,
    ) -> Iterator[torch.Tensor]:
        for _chunk_idx, mask_logits in self._materializer.chunks(query_indices, chunk_size):
            yield (mask_logits > 0).to(device)


class DirectMaskSource(MaskSource):
    """Direct binary-mask source (e.g: SAM2 AMG): chunk a precomputed bool tensor.

    ``pred_masks`` is ``(N, D, H, W)`` already-binarized masks at the model's
    output (processed) resolution. ``query_indices`` indexes the leading object
    axis directly. Evaluation happens at the processed resolution, so masks are
    chunked as-is (no resize).
    """

    def __init__(self, pred_masks: torch.Tensor) -> None:
        self._masks = pred_masks

    def binary_mask_chunks(
        self,
        query_indices: torch.Tensor,
        chunk_size: int,
        device: torch.device,
    ) -> Iterator[torch.Tensor]:
        # Index on the masks' own device, then move the (small) bool chunk to
        # `device` for the pairwise IoU so a CPU-vs-CUDA mismatch can't crash.
        src_device = self._masks.device
        idx = query_indices.to(src_device)
        n = int(idx.numel())
        for start in range(0, n, chunk_size):
            chunk_idx = idx[start : start + chunk_size]
            yield self._masks[chunk_idx].to(device).bool()


def build_mask_source(
    sample: Mapping[str, Any],
    target_size: Sequence[int],
    kind: Optional[MaskSourceKind] = None,
) -> MaskSource:
    """Build the mask source DECLARED by the model for an instance-seg sample.

    The model states which source it produces via ``sample["mask_source"]``
    (``"direct"`` -> :class:`DirectMaskSource`, ``"query"`` -> :class:`QueryMaskSource`).
    ``kind`` overrides the sample key, mainly for tests. The rank/dtype checks are
    contract assertions on the declared kind, NOT the routing mechanism.

    Raises:
        ValueError: when ``mask_source`` is absent or unrecognized, when the keys
            the declared kind requires are missing, or when ``pred_masks`` is
            floating / not rank-4.
    """
    kind = kind if kind is not None else sample.get("mask_source")
    if kind is None:
        raise ValueError(
            "predict_for_eval sample is missing the required 'mask_source' key "
            "(expected 'direct' or 'query'). Models must declare which mask source "
            "they produce; see SAM2.predict_for_eval / MaskDINO.predict_for_eval."
        )

    if kind == "direct":
        pred_masks = sample.get("pred_masks")
        if pred_masks is None:
            raise ValueError(
                "mask_source='direct' requires 'pred_masks' in the predict_for_eval sample."
            )
        if torch.is_floating_point(pred_masks):
            raise ValueError(
                "Instance evaluation requires hard (bool/int) per-instance masks, "
                "but `pred_masks` is floating (soft probabilities). Route soft "
                "semantic maps to the SemanticSegmentationEvaluator instead."
            )
        if pred_masks.dim() != 4:
            raise ValueError(
                "DirectMaskSource expects `pred_masks` of shape (N, D, H, W); "
                f"got {tuple(pred_masks.shape)} (ndim={pred_masks.dim()}). 4D "
                "(T,Z,Y,X,...) instance masks are not supported by the instance "
                "IoU path."
            )
        return DirectMaskSource(pred_masks)

    if kind == "query":
        if (
            sample.get("mask_embeddings") is None
            or sample.get("pixel_decoder_output") is None
        ):
            raise ValueError(
                "mask_source='query' requires both 'mask_embeddings' and "
                "'pixel_decoder_output' in the predict_for_eval sample."
            )
        return QueryMaskSource(
            mask_embeddings=sample["mask_embeddings"],
            pixel_decoder_output=sample["pixel_decoder_output"],
            target_size=target_size,
        )

    raise ValueError(
        f"Unknown mask_source {kind!r} in predict_for_eval sample; "
        "expected 'direct' or 'query'."
    )
