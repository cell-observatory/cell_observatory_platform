import pytest

from cell_observatory_platform.data.data_types import DataKind, kind_family


def test_exact_kinds_resolve_to_themselves():
    assert kind_family("dense") is DataKind.DENSE
    assert kind_family("boxes") is DataKind.BOXES
    assert kind_family("instance_masks") is DataKind.INSTANCE_MASKS


def test_semantic_masks_is_a_family():
    # any semantic_masks_<name> collapses to the SEMANTIC_MASKS family
    assert kind_family("semantic_masks") is DataKind.SEMANTIC_MASKS
    assert kind_family("semantic_masks_membrane") is DataKind.SEMANTIC_MASKS
    assert kind_family("semantic_masks_nucleus") is DataKind.SEMANTIC_MASKS


def test_instance_masks_family_suffix():
    assert kind_family("instance_masks_cells") is DataKind.INSTANCE_MASKS


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        kind_family("not_a_kind")
