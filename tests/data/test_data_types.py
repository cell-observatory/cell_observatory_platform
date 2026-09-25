import pytest
import torch

from cell_observatory_platform.data.data_types import (
    DataKind,
    get_role,
    kind_family,
    parse_annotations_metadata,
)


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


def test_get_role_returns_the_stored_tensor():
    """get_role is a plain Form-D read: the stored tensor comes back by identity."""
    t = torch.zeros(2, 3)
    assert get_role({"denoising": t}, "denoising") is t


def test_get_role_missing_names_both_sides():
    """A missing role raises a KeyError that names the requested role AND the
    roles actually present, so a role-config drift is diagnosable from the message."""
    with pytest.raises(KeyError, match=r"'recon' not found.*\['denoising', 'boundary'\]"):
        get_role({"denoising": torch.zeros(1), "boundary": torch.zeros(1)}, "recon")


# --------------------------------------------------------------------------- #
# parse_annotations_metadata: window-local keys.
#
# Keys are str(timepoint - time_start), so a time_size == 1 row always keys its
# single bucket "0" no matter where it sits in the tile. The parser reads one
# bucket: per-sample targets carry no time axis.
# --------------------------------------------------------------------------- #

def _window(*ids_per_frame):
    """Payload with one bucket per frame, keyed window-locally."""
    return {
        str(t): {
            "instance": [{"local_segmentation_id": i,
                          "bbox_zyxzyx": [0, 0, 0, 1, 1, 1]} for i in ids],
            "semantic": [],
        }
        for t, ids in enumerate(ids_per_frame)
    }


def test_reads_the_requested_window_local_bucket():
    instance, semantic = parse_annotations_metadata(_window([1, 2]))
    assert [leaf["local_segmentation_id"] for leaf in instance] == [1, 2]
    assert semantic == []


def test_offset_selects_among_multiple_buckets():
    payload = _window([1], [2], [3])
    for offset, expected in enumerate([1, 2, 3]):
        instance, _ = parse_annotations_metadata(payload, window_offset=offset)
        assert [leaf["local_segmentation_id"] for leaf in instance] == [expected]


def test_missing_bucket_is_no_objects_not_an_error():
    """A missing key means "no objects in that box"."""
    assert parse_annotations_metadata(_window([1]), window_offset=3) == ([], [])


def test_empty_payloads_parse_to_nothing():
    for raw in (None, "null", "", {}, b"{}"):
        assert parse_annotations_metadata(raw) == ([], [])


def test_a_bare_list_is_rejected():
    """The legacy shape was a flat list of leaves; the time-keyed object replaced
    it, and a silent misread would produce empty targets."""
    with pytest.raises(ValueError, match="time-keyed JSON object"):
        parse_annotations_metadata([{"local_segmentation_id": 1}])
