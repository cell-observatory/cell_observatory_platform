"""
InferenceVisualizer: pure component that turns model outputs into human-consumable visual artifacts.

Handler dispatch is driven by output_type.viz.handler. Used by VizWorkers (Phase 3+) and post-hoc tools.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

_PATH_TOKEN = re.compile(
    r"""
    ([^.[]+)
    (?:\[(\d+)\])?
""",
    re.X,
)


def resolve_path(root: Any, path: str) -> Any:
    """
    Resolve 'path' against 'root'.
    Supports dict keys, attributes, and list indices via '[i]' or bare '.i'.
    Examples:
    - "data_tensor"
    - "metainfo.masks[0]"
    - "metainfo.masks.0"   (treated like index 0)
    """
    cur = root
    for part in path.split("."):
        if part == "":
            continue

        # allow bare numeric segments as list indices (".0")
        if part.isdigit():
            cur = cur[int(part)]
            continue

        # support "key" and "key[i]"
        m = _PATH_TOKEN.fullmatch(part)
        if not m:
            raise KeyError(f"Bad path segment: {part!r} (full path: {path!r})")
        key, idx = m.group(1), m.group(2)

        # descend by key/attr
        if isinstance(cur, dict):
            cur = cur[key]
        else:
            cur = getattr(cur, key)

        # optional index
        if idx is not None:
            cur = cur[int(idx)]

    return cur


def _softmax_last_axis(arr: np.ndarray) -> np.ndarray:
    """Apply softmax along the last axis. Numerically stable."""
    x = np.asarray(arr, dtype=np.float64)
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True)).astype(arr.dtype if hasattr(arr, "dtype") else np.float32)


class InferenceVisualizer:
    """
    Pure component that dispatches to visualization handlers based on output_type.viz.handler.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, Callable[..., None]] = {}
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register built-in handlers for viz.handler names."""
        self._handlers["semantic_map"] = self._handle_semantic_map
        self._handlers["instance_overlay"] = self._handle_instance_overlay
        self._handlers["pca"] = self._handle_feature_viz
        self._handlers["patch_cosine"] = self._handle_feature_viz
        self._handlers["bbox_overlay"] = self._handle_bbox_overlay
        self._handlers["logits_to_probs"] = self._handle_logits_to_probs

    def _handle_semantic_map(
        self,
        output_name: str,
        output_type_cfg: Dict[str, Any],
        data: np.ndarray,
        context: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        from cell_observatory_platform.inference.utils import save_semantic_predictions

        name = context.get("identifier") or context.get("identifiers", [output_name])
        if isinstance(name, (list, tuple)):
            name = name[0] if name else output_name
        save_dir = context.get("save_dir", kwargs.get("save_dir", "."))
        image = context.get("image")
        if image is None:
            raise ValueError("semantic_map handler requires context['image']")
        save_semantic_predictions(
            name=str(name),
            pred_semantic=data,
            image=image,
            save_dir=save_dir,
            save_as_volume=context.get("save_as_volume", False),
            save_as_pdf=context.get("save_as_pdf", True),
            z_step_pdf=context.get("z_step_pdf", 1),
            filetype=context.get("filetype", "zarr"),
            num_classes=context.get("num_classes"),
            outputs_metadata=context.get("outputs_metadata"),
            gt_semantic=context.get("gt_semantic"),
            targets=context.get("targets"),
            data_sample=context.get("data_sample"),
            batch_idx=context.get("batch_idx", 0),
            auxiliary_outputs=context.get("auxiliary_outputs"),
            resolve_path_func=resolve_path,
            zarr_chunk_shape=context.get("zarr_chunk_shape"),
            zarr_shard_shape=context.get("zarr_shard_shape"),
            pmin=context.get("pmin", 1.0),
            pmax=context.get("pmax", 99.0),
            mip_depth=context.get("mip_depth", 20),
            class_names=context.get("class_names"),
            main_output_name=context.get("main_output_name", output_name),
        )

    def _handle_instance_overlay(
        self,
        output_name: str,
        output_type_cfg: Dict[str, Any],
        data: np.ndarray,
        context: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        from cell_observatory_platform.inference.utils import save_instance_predictions

        save_dir = context.get("save_dir", kwargs.get("save_dir", "."))
        identifiers = context.get("identifiers")
        images = context.get("images")
        preds = context.get("preds")
        if identifiers is None or images is None or preds is None:
            raise ValueError(
                "instance_overlay handler requires context['identifiers'], 'images', 'preds'"
            )
        save_instance_predictions(
            save_dir=save_dir,
            identifiers=identifiers,
            images=images,
            preds=preds,
            targets=context.get("targets"),
            regions=context.get("regions"),
            pred_boxes_format=context.get("pred_boxes_format", "xyzxyz"),
            gt_boxes_format=context.get("gt_boxes_format", "cxcyczwhd"),
            z_step=context.get("z_step", 10),
            pmin=context.get("pmin", 1.0),
            pmax=context.get("pmax", 99.0),
            scale_gt_boxes=context.get("scale_gt_boxes", True),
            input_format=context.get("input_format", "ZYXC"),
            ortho=context.get("ortho", False),
        )

    def _handle_feature_viz(
        self,
        output_name: str,
        output_type_cfg: Dict[str, Any],
        data: np.ndarray,
        context: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        from cell_observatory_platform.inference.utils import save_feature_visualizations

        viz_cfg = output_type_cfg.get("viz") or {}
        handler_name = viz_cfg.get("handler", "pca")
        viz_mode = "patch_cosine" if handler_name == "patch_cosine" else "pca"
        predictions = context.get("predictions")
        if predictions is None:
            predictions = {context.get("gt_key", "data_tensor"): context.get("image"), output_name: data}
        elif output_name not in predictions:
            predictions = dict(predictions)
            predictions[output_name] = data
        name = context.get("identifier") or output_name
        if isinstance(name, (list, tuple)):
            name = name[0] if name else output_name
        save_feature_visualizations(
            name=str(name),
            predictions=predictions,
            save_dir=context.get("save_dir", "."),
            gt_key=context.get("gt_key", "data_tensor"),
            feat_key=output_name,
            z_step_pdf=context.get("z_step_pdf", 8),
            pmin=context.get("pmin", 1.0),
            pmax=context.get("pmax", 99.0),
            viz=viz_mode,
            stride_zyx=context.get("stride_zyx", (1, 1, 1)),
        )

    def _handle_bbox_overlay(
        self,
        output_name: str,
        output_type_cfg: Dict[str, Any],
        data: np.ndarray,
        context: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        from cell_observatory_platform.inference.utils import save_bbox_overlay

        image = context.get("image")
        save_dir = context.get("save_dir", kwargs.get("save_dir", "."))
        identifier = context.get("identifier") or output_name
        if isinstance(identifier, (list, tuple)):
            identifier = identifier[0] if identifier else output_name
        if image is None:
            raise ValueError("bbox_overlay handler requires context['image']")
        save_bbox_overlay(
            pred_boxes_xyzxyz=data,
            image=image,
            save_dir=save_dir,
            identifier=str(identifier),
            **{k: v for k, v in context.items() if k in ("z_step", "pmin", "pmax")},
        )

    def _handle_logits_to_probs(
        self,
        output_name: str,
        output_type_cfg: Dict[str, Any],
        data: np.ndarray,
        context: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        viz_cfg = output_type_cfg.get("viz") or {}
        delegate_to = viz_cfg.get("delegate_to", "semantic_map")
        probs = _softmax_last_axis(data)
        delegate_cfg = dict(output_type_cfg)
        if "viz" not in delegate_cfg:
            delegate_cfg["viz"] = {}
        delegate_cfg["viz"] = dict(delegate_cfg["viz"])
        delegate_cfg["viz"]["handler"] = delegate_to
        self.visualize(
            output_name=output_name,
            output_type_cfg=delegate_cfg,
            data=probs,
            context=context,
            **kwargs,
        )

    def visualize(
        self,
        output_name: str,
        output_type_cfg: Dict[str, Any],
        data: np.ndarray,
        context: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """
        Dispatch to the appropriate handler based on output_type.viz.handler.
        """
        viz_cfg = output_type_cfg.get("viz") or {}
        handler_name = viz_cfg.get("handler")
        if not handler_name:
            raise ValueError(
                f"output_type {output_type_cfg.get('output_type', 'unknown')} has no viz.handler"
            )
        if handler_name not in self._handlers:
            raise ValueError(
                f"Unknown viz.handler: {handler_name}. Registered: {list(self._handlers.keys())}"
            )
        self._handlers[handler_name](
            output_name=output_name,
            output_type_cfg=output_type_cfg,
            data=data,
            context=context,
            **kwargs,
        )
