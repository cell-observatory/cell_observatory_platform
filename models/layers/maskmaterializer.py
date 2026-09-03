from typing import Iterator, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F


class MaskMaterializer:
    """Chunked, per-image mask materializer for MaskDINO.

    Args:
        mask_embeddings: ``(Q, mask_dim)`` projected query embeddings for one
            image (last decoder layer). May be in any floating dtype.
        pixel_decoder_output: ``(mask_dim, d, h, w)`` finest pixel-decoder
            feature map for the same image.
        target_size: ``(D, H, W)`` spatial size to upsample masks to (typically
            the original input image size).
        chunk_size: maximum number of queries to materialize simultaneously.
        upsample_dtype: dtype used inside the trilinear interpolate. Defaults
            to ``torch.float32`` because trilinear in fp16 has noticeable
            artefacts on small structures.

    Chunked mask materialization for MaskDINO-style detectors.

    MaskDINO produces ``num_queries`` object queries that, at the final decoder
    layer, project to a ``mask_dim``-vector each. To turn each query into a 3D
    binary mask the model evaluates::

        low_res_mask[b, q] = einsum("bqc, bcdhw->bqdhw", mask_embeddings, pixel_decoder_output)
        full_res_mask[b, q] = F.interpolate(low_res_mask[b, q], size=(D, H, W), mode="trilinear")

    The interpolation step is the memory bottleneck for large 3D volumes: at
    ``D = H = W = 256`` and ``topk_per_image = 100``, a single batch element costs
    ``100 * 256**3 * 4 B = 6.7 GB`` in fp32 (3.3 GB in fp16) just for the mask
    logits — before binarization, IoU computation, etc.

    ``MaskMaterializer`` offers a single entry point — ``chunks(query_indices)`` —
    that yields the upsampled mask logits for ``chunk_size`` queries at a time.
    Callers (``MaskDINO.predict``, ``InstanceSegmentationEvaluator``) consume one
    chunk, free it, and ask for the next. Peak memory drops from
    ``topk * D*H*W`` to ``chunk_size * D*H*W``.

    Notes:
        * The chunked einsum is identical to the global one because the einsum is
        parallelizable along the query axis.
        * Per-chunk logits are produced in the model's compute dtype (typically
        bf16/fp16 when AMP is active); upsample is forced to fp32 internally to
        avoid trilinear precision issues, then cast back. Callers that need a
        different dtype should cast on consumption.
        * Bbox-cropped materialization is not implemented yet but is a natural
        extension: take the union of ``pred_box`` and ``gt_box``, crop the
        pixel-decoder feature map to the corresponding low-res region, and only
        upsample within that crop. We expose enough state (the raw embeddings)
        to add this without changing callers.
    """

    def __init__(
        self,
        mask_embeddings: torch.Tensor,
        pixel_decoder_output: torch.Tensor,
        target_size: Sequence[int],
        chunk_size: int = 8,
        upsample_dtype: torch.dtype = torch.float32,
    ) -> None:
        if mask_embeddings.dim() != 2:
            raise ValueError(
                f"mask_embeddings must be (Q, mask_dim); got {tuple(mask_embeddings.shape)}"
            )
        if pixel_decoder_output.dim() != 4:
            raise ValueError(
                f"pixel_decoder_output must be (mask_dim, d, h, w); got "
                f"{tuple(pixel_decoder_output.shape)}"
            )
        if mask_embeddings.shape[1] != pixel_decoder_output.shape[0]:
            raise ValueError(
                "mask_dim mismatch: mask_embeddings has "
                f"{mask_embeddings.shape[1]} channels but pixel_decoder_output has "
                f"{pixel_decoder_output.shape[0]} channels"
            )
        if len(target_size) != 3:
            raise ValueError(f"target_size must be 3D; got {tuple(target_size)}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive; got {chunk_size}")

        self.mask_embeddings = mask_embeddings
        self.pixel_decoder_output = pixel_decoder_output
        self.target_size = tuple(int(s) for s in target_size)
        self.chunk_size = int(chunk_size)
        self.upsample_dtype = upsample_dtype

    @property
    def num_queries(self) -> int:
        return int(self.mask_embeddings.shape[0])

    @property
    def mask_dim(self) -> int:
        return int(self.mask_embeddings.shape[1])

    @torch.no_grad()
    def materialize(self, query_indices: torch.Tensor) -> torch.Tensor:
        """Materialize the upsampled mask logits for ``query_indices`` at once.

        Use sparingly — defeats the purpose of chunking when the index set is
        large. Provided for parity with the legacy non-chunked path.
        """
        if query_indices.numel() == 0:
            return self.mask_embeddings.new_zeros((0, *self.target_size))
        embed = self.mask_embeddings.index_select(0, query_indices)
        return self._materialize_from_embeds(embed)

    @torch.no_grad()
    def chunks(
        self,
        query_indices: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield ``(chunk_query_indices, upsampled_mask_logits)`` per chunk.

        ``upsampled_mask_logits`` has shape ``(k, D, H, W)`` where ``k`` is the
        number of queries in this chunk (last chunk may be smaller). Logits are
        in the model's mask-embedding dtype (no sigmoid applied).
        """
        if query_indices.numel() == 0:
            return
        cs = int(chunk_size) if chunk_size is not None else self.chunk_size
        if cs <= 0:
            raise ValueError(f"chunk_size must be positive; got {cs}")
        n = int(query_indices.numel())
        for start in range(0, n, cs):
            end = min(start + cs, n)
            chunk_idx = query_indices[start:end]
            embed = self.mask_embeddings.index_select(0, chunk_idx)
            yield chunk_idx, self._materialize_from_embeds(embed)

    def _materialize_from_embeds(self, mask_embeds_chunk: torch.Tensor) -> torch.Tensor:
        """Run einsum + trilinear upsample for one chunk of queries.

        ``mask_embeds_chunk``: ``(k, mask_dim)``.
        Returns: ``(k, D, H, W)`` logits at ``self.target_size``.
        """
        # Low-res einsum: (k, mask_dim) @ (mask_dim, d*h*w) -> (k, d, h, w).
        # Using matmul on the flattened pixel decoder is slightly cheaper than
        # einsum and easier to reason about for memory.
        c, d, h, w = self.pixel_decoder_output.shape
        low_res = mask_embeds_chunk @ self.pixel_decoder_output.reshape(c, d * h * w)
        low_res = low_res.reshape(-1, d, h, w)  # (k, d, h, w)

        # Trilinear upsample: F.interpolate expects (N, C, D, H, W); treat each
        # query as its own "batch" item with C=1 so memory scales linearly.
        original_dtype = low_res.dtype
        low_res = low_res.to(self.upsample_dtype)
        upsampled = F.interpolate(
            low_res.unsqueeze(1),
            size=self.target_size,
            mode="trilinear",
            align_corners=False,
        ).squeeze(1)  # (k, D, H, W)
        return upsampled.to(original_dtype)
