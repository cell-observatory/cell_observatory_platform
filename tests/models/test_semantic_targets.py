"""Semantic target packaging: the config-declared class taxonomy.

Exercises the packaging block of SemanticSegmentationPreprocessor.forward in
isolation -- a full preprocessor needs a DB + Ray, but the packaging is pure
tensor code, so it is factored into a module-level helper and tested here.

``build_semantic_targets`` is the ONE form conversion in the target contract
(data/data_types.py). What changed with the new DB schema is where the classes
come from:

  BEFORE  one DB channel per class, published as Form-D roles, stacked here.
  NOW     ONE integer channel carrying every class, plus a legend
          (``local_segmentation_id -> object_type_id``) in annotations_metadata;
          this one-hots it. Transform-published roles (boundary/foreground) still
          exist and are still selectable, so both sources are supported.

The final assertions run the packaged targets through the evaluator's own GT
derivation, which is the thing that was broken end-to-end.
"""

import pytest
import torch

import cell_observatory_platform.evaluation.evaluate_postprocess as ep
from cell_observatory_platform.models.layers.preprocessor import build_semantic_targets

B, SPATIAL = 1, (2, 4, 4)

# api.object_types: id -> nk. The annotation leaves carry only the id.
OBJECT_TYPES = {1: "membrane", 2: "cytosol"}


def _legend(*pairs):
    """``semantic`` leaves: (local_segmentation_id, object_type_id)."""
    return [
        {"local_segmentation_id": value, "object_type_id": type_id, "object_subtype_ids": []}
        for value, type_id in pairs
    ]


def _pack(targets, semantic_classes, *, semantic_map=None, legend=()):
    return build_semantic_targets(
        targets,
        semantic_classes,
        semantic_map=semantic_map,
        semantic_legend=list(legend),
        object_type_names=OBJECT_TYPES,
        batch_size=B,
        spatial=SPATIAL,
        device=torch.device("cpu"),
    )


def _semantic_map():
    """One squashed integer channel: 0 background, 1 membrane, 2 cytosol."""
    m = torch.zeros(B, *SPATIAL, dtype=torch.int32)
    m[0, 0, 0, 0] = 1
    m[0, 0, 0, 1] = 2
    return m


def _derived():
    """Transform-published binary roles, unchanged by the schema."""
    boundary = torch.zeros(B, *SPATIAL, dtype=torch.int32)
    boundary[0, 0, 0, 0] = 1
    foreground = torch.zeros(B, *SPATIAL, dtype=torch.int32)
    foreground[0, 0, 0, 1] = 1
    return {"boundary": boundary, "foreground": foreground}


# --------------------------------------------------------------------------- #
# legend classes (the new source)
# --------------------------------------------------------------------------- #


def test_all_resolves_the_legend():
    out, classes = _pack(
        {}, "all", semantic_map=_semantic_map(), legend=_legend((1, 1), (2, 2))
    )
    assert classes == ["membrane", "cytosol"]
    assert out[0]["masks"].shape == (2, *SPATIAL)


def test_one_hot_comes_from_the_labelmap_value():
    out, _ = _pack(
        {}, ["membrane"], semantic_map=_semantic_map(), legend=_legend((1, 1))
    )
    masks = out[0]["masks"]
    assert masks[0, 0, 0, 0].item() is True
    assert masks[0, 0, 0, 1].item() is False  # that voxel is cytosol (value 2)


def test_class_index_is_list_position_not_labelmap_value():
    """The config owns the taxonomy; the DB integer is only a lookup key."""
    out, classes = _pack(
        {},
        ["cytosol", "membrane"],
        semantic_map=_semantic_map(),
        legend=_legend((1, 1), (2, 2)),
    )
    assert classes == ["cytosol", "membrane"]
    assert out[0]["labels"].tolist() == [0, 1]
    # class 0 is cytosol -> the voxel holding value 2
    assert out[0]["masks"][0, 0, 0, 1].item() is True


def test_legend_classes_are_mutually_exclusive():
    """A squashed integer channel cannot overlap, which is what makes the
    evaluator's last-write-wins scatter exact for these classes."""
    out, _ = _pack(
        {}, "all", semantic_map=_semantic_map(), legend=_legend((1, 1), (2, 2))
    )
    assert int(out[0]["masks"].sum(dim=0).max().item()) <= 1


def test_background_cannot_become_a_class():
    """Background is simply absent from the legend -- no config knob needed."""
    _out, classes = _pack(
        {}, "all", semantic_map=_semantic_map(), legend=_legend((1, 1), (2, 2))
    )
    assert "background" not in classes


def test_unknown_object_type_id_raises():
    with pytest.raises(KeyError, match="object_type_id"):
        _pack({}, "all", semantic_map=_semantic_map(), legend=_legend((1, 99)))


def test_legend_class_without_semantic_channel_raises():
    with pytest.raises(ValueError, match="no semantic channel"):
        _pack({}, ["membrane"], semantic_map=None, legend=_legend((1, 1)))


# --------------------------------------------------------------------------- #
# derived roles (unchanged source) and the mix
# --------------------------------------------------------------------------- #


def test_explicit_list_selects_derived_roles():
    out, classes = _pack(_derived(), ["boundary", "foreground"])
    assert classes == ["boundary", "foreground"]
    assert out[0]["masks"].shape == (2, *SPATIAL)


def test_mixed_legend_and_derived_classes():
    out, classes = _pack(
        _derived(),
        ["membrane", "boundary"],
        semantic_map=_semantic_map(),
        legend=_legend((1, 1)),
    )
    assert classes == ["membrane", "boundary"]
    assert out[0]["masks"].shape == (2, *SPATIAL)


def test_unknown_class_names_list_both_sources():
    with pytest.raises(KeyError) as excinfo:
        _pack(_derived(), ["typo_role"], semantic_map=_semantic_map(), legend=_legend((1, 1)))
    message = str(excinfo.value)
    assert "legend classes" in message and "derived roles" in message


def test_no_dense_map_is_stored():
    out, _ = _pack(_derived(), ["boundary", "foreground"])
    assert set(out[0]) == {"masks", "labels"}


def test_output_is_form_s_length_b():
    out, _ = _pack(_derived(), ["boundary", "foreground"])
    assert isinstance(out, list) and len(out) == B


def test_empty_taxonomy_yields_empty_masks():
    out, classes = _pack({}, "all")
    assert classes == []
    assert out[0]["masks"].shape == (0, *SPATIAL)
    assert out[0]["labels"].numel() == 0


# --------------------------------------------------------------------------- #
# end-to-end through the evaluator's GT derivation
# --------------------------------------------------------------------------- #


def test_derived_gt_map_has_no_phantom_class():
    out, classes = _pack(_derived(), ["boundary", "foreground"])
    gt = ep.gt_semantic_map(out[0], SPATIAL, source="masks")
    assert set(gt.unique().tolist()) <= {0, 1, 2}
    assert len(classes) == 2


def test_legend_gt_map_round_trips():
    out, classes = _pack(
        {}, "all", semantic_map=_semantic_map(), legend=_legend((1, 1), (2, 2))
    )
    gt = ep.gt_semantic_map(out[0], SPATIAL, source="masks")
    # class + 1, background 0
    assert gt[0, 0, 0].item() == 1
    assert gt[0, 0, 1].item() == 2
    assert gt[1, 0, 0].item() == 0
    assert len(classes) == 2
