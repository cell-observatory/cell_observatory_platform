import pytest

import torch

from cell_observatory_platform.data.data_shapes import MULTICHANNEL_HYPERCUBE

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def make_tensor(shape):
    """Create a tensor whose values are a simple ascending range so that
    permutations can be compared for exact equality."""
    numel = int(torch.tensor(shape).prod().item())
    return torch.arange(numel, dtype=torch.int64).reshape(shape)

def cf_variant(layout):
    if layout.is_3d():
        return MULTICHANNEL_HYPERCUBE.CZYX if not layout.has_temporal_dim() else MULTICHANNEL_HYPERCUBE.CTYX
    elif layout.is_4d():
        return MULTICHANNEL_HYPERCUBE.CTZYX
    raise TypeError

def cl_variant(layout):
    if layout.is_3d():
        return MULTICHANNEL_HYPERCUBE.ZYXC if not layout.has_temporal_dim() else MULTICHANNEL_HYPERCUBE.TYXC
    elif layout.is_4d():
        return MULTICHANNEL_HYPERCUBE.TZYXC
    raise TypeError

# -----------------------------------------------------------------------------
# tables describing every legal layout, with and without batch dim
# -----------------------------------------------------------------------------

CASES_3D = [
    # enum, shape (no batch), shape (with batch), has_temporal
    (MULTICHANNEL_HYPERCUBE.CZYX, (3, 4, 5, 6), (2, 3, 4, 5, 6), False),
    (MULTICHANNEL_HYPERCUBE.ZYXC, (4, 5, 6, 3), (2, 4, 5, 6, 3), False),
    (MULTICHANNEL_HYPERCUBE.CTYX, (3, 7, 5, 6), (2, 3, 7, 5, 6), True),
    (MULTICHANNEL_HYPERCUBE.TYXC, (7, 5, 6, 3), (2, 7, 5, 6, 3), True),
]

CASES_4D = [
    (MULTICHANNEL_HYPERCUBE.CTZYX, (3, 7, 4, 5, 6), (2, 3, 7, 4, 5, 6), True),
    (MULTICHANNEL_HYPERCUBE.TZYXC, (7, 4, 5, 6, 3), (2, 7, 4, 5, 6, 3), True),
]

# -----------------------------------------------------------------------------
# generic property tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("layout, shape_nb, shape_b, has_t", CASES_3D + CASES_4D)
def test_axes_property(layout, shape_nb, shape_b, has_t):
    assert isinstance(layout.axes, tuple)
    assert "".join(layout.axes) == layout.value


@pytest.mark.parametrize("layout, shape_nb, shape_b, has_t", CASES_3D)
def test_channel_first_last_flags_3d(layout, shape_nb, shape_b, has_t):
    # exactly one of the two flags must be True
    assert layout.is_channel_first() ^ layout.is_channel_last()
    assert layout.is_channel_first() == (layout in (MULTICHANNEL_HYPERCUBE.CZYX, MULTICHANNEL_HYPERCUBE.CTYX))
    assert layout.is_channel_last() == (layout in (MULTICHANNEL_HYPERCUBE.ZYXC, MULTICHANNEL_HYPERCUBE.TYXC))
    # temporal flag must match the design table above
    assert layout.has_temporal_dim() == (layout in (MULTICHANNEL_HYPERCUBE.CTYX, MULTICHANNEL_HYPERCUBE.TYXC))


@pytest.mark.parametrize("layout, shape_nb, shape_b, has_t", CASES_4D)
def test_channel_first_last_flags_4d(layout, shape_nb, shape_b, has_t):
    assert layout.is_channel_first() ^ layout.is_channel_last()
    assert layout.is_channel_first() == (layout is MULTICHANNEL_HYPERCUBE.CTZYX)
    assert layout.is_channel_last() == (layout is MULTICHANNEL_HYPERCUBE.TZYXC)
    assert layout.has_temporal_dim()

# -----------------------------------------------------------------------------
# get_image_shape_tuple and get_image_shape_dict
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("layout, shape_nb, shape_b, has_t", CASES_3D)
def test_image_shape_helpers_3d(layout, shape_nb, shape_b, has_t):
    for shape in (shape_nb, shape_b):
        tensor = make_tensor(shape)
        tup = layout.get_image_shape_tuple(tensor)
        d = layout.get_image_shape_dict(tensor)

        assert set(tup) == {v for k, v in d.items() if k != "c"}

        keys_expected = {"c", "y", "x"} | ({"z"} if not has_t else {"t"})
        assert keys_expected.issubset(d.keys())
        if not has_t:
            assert d["z"] == (shape[-3] if layout.is_channel_first() else shape[0 if len(shape)==4 else 1])
        else:
            assert d["t"] == (shape[-3] if layout.is_channel_first() else shape[0 if len(shape)==4 else 1])
        
        if len(shape) == 4:
            assert d["c"] == (shape[0] if layout.is_channel_first() else shape[-1])
        else:
            assert d["c"] == (shape[1] if layout.is_channel_first() else shape[-1])


@pytest.mark.parametrize("layout, shape_nb, shape_b, has_t", CASES_4D)
def test_image_shape_helpers_4d(layout, shape_nb, shape_b, has_t):
    for shape in (shape_nb, shape_b):
        tensor = make_tensor(shape)
        tup = layout.get_image_shape_tuple(tensor)
        d  = layout.get_image_shape_dict(tensor)

        # Tuple, dict consistency
        assert set(tup) == {d["t"], d["z"], d["y"], d["x"]}

        # check individual entries
        if layout.is_channel_first():
            if len(shape) == 5:
                c, t = shape[0], shape[1]
            else:
                c, t = shape[1], shape[2]
        else:
            c, t = shape[-1], shape[0 if len(shape)==5 else 1]
        
        assert d["c"] == c
        assert d["t"] == t

# ----------------------------------------------------------------------------- 
# spatial / temporal helpers
# ----------------------------------------------------------------------------- 

@pytest.mark.parametrize("layout, shape_nb, shape_b, has_t", CASES_3D)
def test_spatial_temporal_shape_3d(layout, shape_nb, shape_b, has_t):
    tensor = make_tensor(shape_nb)

    spatial = layout.get_spatial_shape(tensor)
    assert all(isinstance(v, int) for v in spatial)

    d = layout.get_image_shape_dict(tensor)
    if has_t:
        assert spatial == (d["y"], d["x"])
    else:
        assert spatial == (d["z"], d["y"], d["x"])

    if has_t:
        assert layout.get_temporal_shape(tensor) == d["t"]
    else:
        with pytest.raises(ValueError):
            layout.get_temporal_shape(tensor)

    # same for batch shape
    tensor_b = make_tensor(shape_b)
    spatial_b = layout.get_spatial_shape(tensor_b)
    assert all(isinstance(v, int) for v in spatial_b)

    d_b = layout.get_image_shape_dict(tensor_b)
    if has_t:
        assert spatial_b == (d_b["y"], d_b["x"])
    else:
        assert spatial_b == (d_b["z"], d_b["y"], d_b["x"])  
    
    if has_t:
        assert layout.get_temporal_shape(tensor_b) == d_b["t"]
    else:
        with pytest.raises(ValueError):
            layout.get_temporal_shape(tensor_b)


@pytest.mark.parametrize("layout, shape_nb, shape_b, has_t", CASES_4D)
def test_spatial_temporal_shape_4d(layout, shape_nb, shape_b, has_t):
    tensor = make_tensor(shape_nb)
    z, y, x = layout.get_spatial_shape(tensor)
    assert (z, y, x) == (shape_nb[-3 if layout.is_channel_first() else 1],
                         shape_nb[-2 if layout.is_channel_first() else 2],
                         shape_nb[-1 if layout.is_channel_first() else 3])
    assert layout.get_temporal_shape(tensor) == (shape_nb[1] if layout.is_channel_first() else shape_nb[0])

    tensor_b = make_tensor(shape_b)
    z, y, x = layout.get_spatial_shape(tensor_b)
    assert (z, y, x) == (shape_b[-3 if layout.is_channel_first() else 2],
                         shape_b[-2 if layout.is_channel_first() else 3],
                         shape_b[-1 if layout.is_channel_first() else 4])
    assert layout.get_temporal_shape(tensor_b) == (shape_b[2] if layout.is_channel_first() else shape_b[1])

# -----------------------------------------------------------------------------
# num_channels / num_timepoints
# ----------------------------------------------------------------------------- 

@pytest.mark.parametrize("layout, shape_nb, shape_b, has_t", CASES_3D + CASES_4D)
def test_num_channels_and_timepoints(layout, shape_nb, shape_b, has_t):
    tensor = make_tensor(shape_nb)
    assert layout.num_channels(tensor) == (shape_nb[0] if layout.is_channel_first() else shape_nb[-1])

    tp = layout.num_timepoints(tensor)
    if layout.has_temporal_dim():
        expected = shape_nb[1] if layout.is_channel_first() else shape_nb[0]
        assert tp == expected
    else:
        assert tp is None

    # same for batch shape
    tensor_b = make_tensor(shape_b)
    assert layout.num_channels(tensor_b) == (shape_b[1] if layout.is_channel_first() else shape_b[-1])

    tp_b = layout.num_timepoints(tensor_b)
    if layout.has_temporal_dim():
        expected = shape_b[2] if layout.is_channel_first() else shape_b[1]
        assert tp_b == expected
    else:
        assert tp_b is None

# -----------------------------------------------------------------------------
# permutation round-trip tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("layout, shape_nb, shape_b, has_t", CASES_3D + CASES_4D)
def test_permutation_roundtrip(layout, shape_nb, shape_b, has_t):
    tensor_orig = make_tensor(shape_nb)

    if layout.is_channel_first():
        new_tensor = layout.to_channel_last(tensor_orig)
        # sanity: channel dim size is correct
        assert new_tensor.shape[cl_variant(layout).axes.index("C")] == layout.num_channels(tensor_orig)
        old_tensor = cl_variant(layout).to_channel_first(new_tensor)
        assert torch.equal(tensor_orig, old_tensor), (
            "Round-trip permutation did not preserve data: "
            f"{tensor_orig.shape} vs {old_tensor.shape}"
        )
    else:
        new_tensor = layout.to_channel_first(tensor_orig)
        # sanity: channel dim size is correct
        assert new_tensor.shape[cf_variant(layout).axes.index("C")] == layout.num_channels(tensor_orig)
        old_tensor = cf_variant(layout).to_channel_last(new_tensor)
        assert torch.equal(tensor_orig, old_tensor), (
            "Round-trip permutation did not preserve data: "
            f"{tensor_orig.shape} vs {old_tensor.shape}"
        )

    tensor_orig_b = make_tensor(shape_b)
    if layout.is_channel_first():
        new_tensor = layout.to_channel_last(tensor_orig_b)
        # sanity: channel dim size is correct
        assert new_tensor.shape[1+cl_variant(layout).axes.index("C")] == layout.num_channels(tensor_orig_b)
        old_tensor = cl_variant(layout).to_channel_first(new_tensor)
        assert torch.equal(tensor_orig_b, old_tensor), (
            "Round-trip permutation did not preserve data: "
            f"{tensor_orig_b.shape} vs {old_tensor.shape}"
        )
    else:
        new_tensor = layout.to_channel_first(tensor_orig_b)
        # sanity: channel dim size is correct
        assert new_tensor.shape[1+cf_variant(layout).axes.index("C")] == layout.num_channels(tensor_orig_b)
        old_tensor = cf_variant(layout).to_channel_last(new_tensor)
        assert torch.equal(tensor_orig_b, old_tensor), (
            "Round-trip permutation did not preserve data: "
            f"{tensor_orig_b.shape} vs {old_tensor.shape}"
        )