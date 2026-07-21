"""Semantic target packaging: the config-declared class taxonomy.

Exercises the packaging block of SemanticSegmentationPreprocessor.forward in
isolation -- a full preprocessor needs a DB + Ray, but the packaging is pure tensor
code, so it is factored into a module-level helper and tested here.

The final assertions run the packaged targets through the evaluator's own GT
derivation, which is the thing that was broken end-to-end.
"""

import pytest
import torch

import cell_observatory_platform.evaluation.evaluate_postprocess as ep
from cell_observatory_platform.models.layers.preprocessor import build_semantic_targets

MEMBRANE = "semantic_segmentation_membrane"


def _membrane_with_derived():
    """The current ABC setup: one channel role plus two appended derived maps."""
    stack = torch.zeros(3, 2, 4, 4, dtype=torch.int32)
    stack[0, 0, 0, :2] = 1          # source footprint
    stack[1, 0, 0, 0] = 1           # boundary
    stack[2, 0, 0, 1] = 1           # interior
    return {
        "semantic_maps": stack,
        "semantic_roles": [MEMBRANE, "boundary", "foreground"],
        "channel_roles": [MEMBRANE],
    }


def _three_channels():
    stack = torch.zeros(3, 2, 4, 4, dtype=torch.int32)
    stack[0, 0, 0, 0] = 7           # cytosol   (instance ids are arbitrary)
    stack[1, 0, 0, 1] = 3           # golgi
    stack[2, 0, 0, 2] = 9           # membrane
    roles = ["cytosol", "golgi", "membrane"]
    return {"semantic_maps": stack, "semantic_roles": roles, "channel_roles": list(roles)}


def test_all_selects_channel_roles_only():
    # Derived slices are not channel roles, so "all" gives the single membrane class:
    # foreground/background of the membrane channel.
    out, classes = build_semantic_targets(_membrane_with_derived(), "all")
    assert classes == [MEMBRANE]
    assert out["masks"].shape[0] == 1
    assert out["labels"].tolist() == [0]


def test_explicit_list_selects_derived_slices():
    out, classes = build_semantic_targets(_membrane_with_derived(), ["boundary", "foreground"])
    assert classes == ["boundary", "foreground"]
    assert out["masks"].shape[0] == 2
    assert out["labels"].tolist() == [0, 1]


def test_all_selects_every_channel_role_when_no_transforms_ran():
    out, classes = build_semantic_targets(_three_channels(), "all")
    assert classes == ["cytosol", "golgi", "membrane"]
    assert out["labels"].tolist() == [0, 1, 2]


def test_explicit_list_controls_order_and_may_subset():
    out, classes = build_semantic_targets(_three_channels(), ["membrane", "cytosol"])
    assert classes == ["membrane", "cytosol"]
    assert out["masks"].shape[0] == 2


def test_unknown_role_raises():
    with pytest.raises(KeyError, match="typo_role"):
        build_semantic_targets(_three_channels(), ["typo_role"])


def test_no_dense_map_is_stored():
    """The single-label map is derived at eval time, never carried on the target."""
    out, _ = build_semantic_targets(_three_channels(), "all")
    assert set(out) == {"masks", "labels"}


def test_empty_stack_yields_empty_taxonomy():
    t = {"semantic_maps": torch.zeros(0, 2, 4, 4, dtype=torch.int32),
         "semantic_roles": [], "channel_roles": []}
    out, classes = build_semantic_targets(t, "all")
    assert classes == []
    assert out["masks"].shape == (0, 2, 4, 4)
    assert out["labels"].tolist() == []


# --- integration with the evaluator's GT derivation -------------------------------

def test_derived_gt_map_has_no_phantom_class():
    """The regression: previously value 1 was erased because the source footprint was
    stacked alongside its own boundary/interior partition."""
    out, classes = build_semantic_targets(_membrane_with_derived(), ["boundary", "foreground"])
    gt = ep.gt_semantic_map(out, size=(2, 4, 4), source="masks")

    assert gt[0, 0, 0].item() == 1          # boundary -> class 0 + 1
    assert gt[0, 0, 1].item() == 2          # interior -> class 1 + 1
    assert sorted(torch.unique(gt).tolist()) == [0, 1, 2]   # no gap, no phantom


def test_derived_gt_map_multi_role():
    out, classes = build_semantic_targets(_three_channels(), "all")
    gt = ep.gt_semantic_map(out, size=(2, 4, 4), source="masks")
    assert gt[0, 0, 0].item() == 1          # cytosol
    assert gt[0, 0, 1].item() == 2          # golgi
    assert gt[0, 0, 2].item() == 3          # membrane
    assert gt.dtype == torch.long
