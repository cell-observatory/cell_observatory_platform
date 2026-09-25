"""Channel selection helpers.

Channels arrive as positionally-aligned arrays. Three properties are pinned here:
selection is order-INSENSITIVE (channel layout is a function of the row and the
selected set, never of config list order), mask channels are always retained and
land in the tail, and ``channel_idx`` must be dense so that "array position" and
"zarr C-axis index" are the same number.
"""

import pytest

from cell_observatory_platform.data.datasets.utils import (
    channel_tokens_for_selection,
    remap_channel_roles_to_selection,
    resolve_channel_indices,
)

# The shape every row has: aligned arrays in ascending channel_idx order. The
# XOR constraint on roi_channels forces localization NULL on a mask channel and
# annotation_type NULL on a data channel.
IDX = [0, 1, 2, 3]
TYPES = ["data", "data", "data", "mask"]
LOCS = ["membrane", "cytosol", "cytosol", None]
ANNOS = [None, None, None, "instance"]


# --------------------------------------------------------------------------- #
# resolve_channel_indices
# --------------------------------------------------------------------------- #


def test_selects_requested_data_channels_then_masks():
    assert resolve_channel_indices(IDX, TYPES, LOCS, ["membrane"]) == [0, 3]


def test_one_localization_returns_every_channel_carrying_it():
    assert resolve_channel_indices(IDX, TYPES, LOCS, ["cytosol"]) == [1, 2, 3]


def test_a_repeated_token_does_not_duplicate_channels():
    """A channel appears once however many times its localization is listed --
    a repeat would emit a duplicated channel and a channel_size the buffer
    sizing disagrees with."""
    assert resolve_channel_indices(IDX, TYPES, LOCS, ["cytosol", "cytosol"]) == [1, 2, 3]


def test_selection_order_is_the_source_order_not_the_request_order():
    """selected_channel_localizations is a SET. Driving tensor channel order from
    YAML list order would make two configs requesting the same channels produce
    different tensors -- and silently mismatch a checkpoint trained under the
    other ordering."""
    assert resolve_channel_indices(IDX, TYPES, LOCS, ["cytosol", "membrane"]) == [0, 1, 2, 3]
    assert resolve_channel_indices(IDX, TYPES, LOCS, ["membrane", "cytosol"]) == [0, 1, 2, 3]


def test_selection_depends_only_on_the_requested_set():
    permutations = (
        ["membrane", "cytosol"],
        ["cytosol", "membrane"],
        ["cytosol", "membrane", "cytosol"],   # duplicates collapse
        ["  MEMBRANE ", "CYTOSOL"],           # normalization is order-agnostic too
    )
    outs = {tuple(resolve_channel_indices(IDX, TYPES, LOCS, p)) for p in permutations}
    assert len(outs) == 1, outs


def test_masks_are_always_retained():
    """A localization-only selection can never NAME the labelmap -- roi_channels
    forces localization NULL on a mask channel -- so it would be dropped silently
    and _split_channels would then find no object channels."""
    for requested in (["membrane"], ["cytosol"], ["membrane", "cytosol"]):
        assert 3 in resolve_channel_indices(IDX, TYPES, LOCS, requested)


def test_masks_land_in_the_tail():
    """preprocessor._split_channels requires signal channels to be a contiguous
    prefix with object channels in the tail, and raises otherwise."""
    out = resolve_channel_indices(IDX, TYPES, LOCS, ["cytosol", "membrane"])
    assert out[-1] == 3


def test_sparse_channel_idx_is_rejected_rather_than_guessed():
    """The DB models array POSITION and zarr C-AXIS INDEX as separate things, but
    every row of both training views carries {0,1,2,3,4,5}. Enforcing that keeps
    the loader and preprocessor on ONE numbering scheme; if a sparse array ever
    appears, the selection paths need a deliberate revisit rather than a silent
    read of the wrong channels."""
    with pytest.raises(ValueError, match="dense and ascending"):
        resolve_channel_indices([0, 2, 5, 9], TYPES, LOCS, ["cytosol"])


def test_dense_channel_idx_makes_position_and_index_the_same_number():
    out = resolve_channel_indices(IDX, TYPES, LOCS, ["cytosol"])
    assert out == [1, 2, 3]          # positions 1, 2 (data) then 3 (mask)


def test_no_request_still_orders_masks_last():
    shuffled_types = ["data", "mask", "data", "data"]
    shuffled_annos = [None, "instance", None, None]
    out = resolve_channel_indices(IDX, shuffled_types, [None] * 4, None)
    assert out == [0, 2, 3, 1]


def test_no_request_and_no_mask_loads_everything():
    """None means "load every channel in on-disk order"."""
    assert resolve_channel_indices([0, 1], ["data", "data"], ["a", "b"], None) is None


def test_missing_localization_raises_and_names_the_row():
    with pytest.raises(ValueError, match="not present among data channels"):
        resolve_channel_indices(IDX, TYPES, LOCS, ["golgi"])


def test_every_unknown_localization_is_named_at_once():
    """Report the whole bad set, not just whichever one happened to be first --
    fixing a config one token per run is a slow way to find three typos."""
    with pytest.raises(ValueError) as excinfo:
        resolve_channel_indices(IDX, TYPES, LOCS, ["golgi", "membrane", "gfp"])
    assert "'gfp'" in str(excinfo.value) and "'golgi'" in str(excinfo.value)


def test_localization_on_a_mask_channel_does_not_match():
    """Selection filters SIGNAL channels only; a mask channel is retained because
    it is a mask, never because its localization matched."""
    locs = ["membrane", "cytosol", "cytosol", "membrane"]  # malformed: mask w/ a loc
    assert resolve_channel_indices(IDX, TYPES, locs, ["membrane"]) == [0, 3]


def test_null_channel_metadata_raises():
    """array_agg over an empty set yields NULL, not an empty array: the ROI has no
    roi_channels rows. Defaulting to channel 0 would be a guess."""
    with pytest.raises(ValueError, match="channel_idx is NULL"):
        resolve_channel_indices(None, None, None, ["membrane"])


def test_misaligned_arrays_raise():
    with pytest.raises(ValueError, match="not aligned"):
        resolve_channel_indices([0, 1, 2], ["data", "data"], LOCS, None)


def test_localization_matching_is_case_and_whitespace_insensitive():
    assert resolve_channel_indices(IDX, TYPES, LOCS, ["  MEMBRANE "]) == [0, 3]


# --------------------------------------------------------------------------- #
# remap_channel_roles_to_selection
# --------------------------------------------------------------------------- #


def test_roles_are_keyed_by_post_selection_position():
    """After slicing, channel k of the emitted tensor is source
    selected[k] -- the preprocessor partitions by those NEW positions."""
    selected = resolve_channel_indices(IDX, TYPES, LOCS, ["membrane"])   # [0, 3]
    assert remap_channel_roles_to_selection(TYPES, ANNOS, IDX, selected) == {
        "1": "instance_masks"
    }


def test_data_channels_get_no_role():
    """No entry means INPUT to partition_channels -- the same meaning the old
    channel_mapping conveyed by carrying a non-object role."""
    roles = remap_channel_roles_to_selection(TYPES, ANNOS, IDX, [0, 1, 2, 3])
    assert set(roles) == {"3"}


def test_annotation_type_becomes_the_datakind_role():
    types = ["data", "mask", "mask"]
    annos = [None, "instance", "semantic"]
    roles = remap_channel_roles_to_selection(types, annos, [0, 1, 2], [0, 1, 2])
    assert roles == {"1": "instance_masks", "2": "semantic_masks"}


def test_mask_without_annotation_type_raises():
    """The DB XOR constraint should make this impossible; if it reaches us the
    role table would silently lose a GT channel."""
    with pytest.raises(ValueError, match="annotation_type"):
        remap_channel_roles_to_selection(["mask"], [None], [0], [0])


def test_none_selection_means_every_channel():
    roles = remap_channel_roles_to_selection(TYPES, ANNOS, IDX, None)
    assert roles == {"3": "instance_masks"}


def test_as_list_parses_json_serialized_lists():
    from cell_observatory_platform.data.datasets.utils import _as_list
    assert _as_list('["data", "data", "mask"]') == ["data", "data", "mask"]
    assert _as_list('[0, 1, 2]') == [0, 1, 2]
    assert _as_list(["a", "b"]) == ["a", "b"]
    assert _as_list(None) is None


def test_json_list_serializes_numpy_scalars_and_nulls():
    import numpy as np
    from cell_observatory_platform.data.datasets.utils import _as_list, _json_list
    idx = np.array([0, 1, 2], dtype=np.int16)
    assert _json_list(idx) == "[0,1,2]" and _as_list(_json_list(idx)) == [0, 1, 2]
    loc = np.array(["membrane", None, "cytosol"], dtype=object)
    assert _as_list(_json_list(loc)) == ["membrane", None, "cytosol"]
    assert _json_list(np.array(["data", "mask"])) == '["data","mask"]'
    assert _json_list(None) == "null" and _as_list("null") is None


# --------------------------------------------------------------------------- #
# channel_tokens_for_selection
# --------------------------------------------------------------------------- #

FLUORS = ["mstaygold", "Electra2", "mTFP1", None]


def test_channel_tokens_follow_post_selection_order_and_null_on_masks():
    """Selecting cytosol only emits [cytosol ch1, cytosol ch2, mask]; the token
    list is keyed by those NEW positions and the mask slot is None."""
    sel = resolve_channel_indices(IDX, TYPES, LOCS, ["cytosol"])
    assert sel == [1, 2, 3]
    toks = channel_tokens_for_selection(TYPES, LOCS, FLUORS, IDX, sel)
    assert toks == [["cytosol", "electra2"], ["cytosol", "mtfp1"], None]


def test_channel_tokens_are_normalized_like_the_db_filters():
    toks = channel_tokens_for_selection(
        TYPES, ["  Membrane ", "CYTOSOL", "cytosol", None], FLUORS, IDX, None
    )
    assert toks[0] == ["membrane", "mstaygold"]
    assert toks[1] == ["cytosol", "electra2"]


def test_channel_tokens_keep_a_null_column_as_none():
    """A NULL fluorophore (free-text row not yet mapped) stays None so the
    consumer's unknown-token policy decides, rather than a silent '' token."""
    toks = channel_tokens_for_selection(TYPES, LOCS, [None, None, None, None], IDX, None)
    assert toks[0] == ["membrane", None]


def test_channel_tokens_none_selection_means_source_order():
    toks = channel_tokens_for_selection(TYPES, LOCS, FLUORS, IDX, None)
    assert len(toks) == 4 and toks[3] is None


def test_channel_tokens_accept_json_serialized_arrays():
    import ujson
    toks = channel_tokens_for_selection(
        ujson.dumps(TYPES), ujson.dumps(LOCS), ujson.dumps(FLUORS), ujson.dumps(IDX), [0, 3]
    )
    assert toks == [["membrane", "mstaygold"], None]
