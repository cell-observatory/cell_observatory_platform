import pytest
import torch

from cell_observatory_platform.data.transforms.make_targets import (
    DeepCopyInputsAsTargets,
    ForegroundMasks,
    InstanceToBoundaryMask,
)

# The DB channel role these transforms derive from in production
# (configs/datasets/preprocessor/semantic_segmentation_preprocessor.yaml:28,33).
ROLE = "semantic_segmentation_membrane"


def _sample(labels: torch.Tensor, role: str = ROLE) -> dict:
    """Wrap a (B, D, H, W) labelmap in the Form-D role dict these transforms
    consume (see data/data_types.py): one batched map per role.

    Mirrors SemanticSegmentationPreprocessor.forward, which is the only
    production producer of this shape.
    """
    return {"metainfo": {"targets": {role: labels.to(torch.int32)}}}


def _derived(out: dict, tag: str) -> torch.Tensor:
    """Read a derived role (e.g. "boundary") back out as (B, D, H, W).

    The transforms publish derived maps as new Form-D roles rather than
    returning a tensor, so assertions read the role back out by name.
    """
    return out["metainfo"]["targets"][tag]


def test_deep_copy_inputs_as_targets_clones_tensor():
    data_tensor = torch.randn(2, 3)
    transform = DeepCopyInputsAsTargets()

    result = transform({"data_tensor": data_tensor})
    clone = result["metainfo"]["targets"]["denoising"]   # Form-D role (default)
    clone.add_(1.0)

    assert torch.equal(result["data_tensor"], data_tensor), "Data tensor is was modified"
    assert not torch.equal(clone, result["data_tensor"]), "Modifying targets changed data tensor"
    assert clone.data_ptr() != result["data_tensor"].data_ptr(), "Targets and data tensor share the same memory"


def test_deep_copy_inputs_as_targets_publishes_configured_role():
    result = DeepCopyInputsAsTargets(role="custom")({"data_tensor": torch.zeros(2, 3)})
    assert list(result["metainfo"]["targets"]) == ["custom"]

# FIXME: I think we want this behavior, but for now the implementation 
# generates targets before data hits the preprocessor (during initial mask generation).
# We should streamline this so that targets are generated cosnsitently, 
# OR just remove this and generate targets wherever it makes sense.
# def test_deep_copy_inputs_as_targets_rejects_existing_targets():
#     transform = DeepCopyInputsAsTargets()
#     data = {"data_tensor": torch.zeros(1), "metainfo": {"targets": [torch.ones(1)]}}
#     with pytest.raises(ValueError):
#         transform(data)


def test_deep_copy_inputs_as_targets_requires_data_tensor():
    transform = DeepCopyInputsAsTargets()

    with pytest.raises(KeyError):
        transform({})


# InstanceToBoundaryMask tests

@pytest.mark.parametrize("connectivity,expected_shifts", [
    (1, 6),   # 6-neighborhood
    (2, 18),  # 18-neighborhood
    (3, 26),  # 26-neighborhood
])
def test_connectivity_initialization(connectivity, expected_shifts):
    transform = InstanceToBoundaryMask(ROLE, connectivity=connectivity)
    assert len(transform.shifts) == expected_shifts, f"connectivity={connectivity} should create {expected_shifts} shifts"
    assert transform.connectivity == connectivity


def test_connectivity_default_is_1():
    transform = InstanceToBoundaryMask(ROLE)
    assert transform.connectivity == 1
    assert len(transform.shifts) == 6


def test_invalid_connectivity_raises_error():
    with pytest.raises(ValueError, match="connectivity must be 1, 2, or 3"):
        InstanceToBoundaryMask(ROLE, connectivity=0)
    with pytest.raises(ValueError, match="connectivity must be 1, 2, or 3"):
        InstanceToBoundaryMask(ROLE, connectivity=4)


def test_boundary_detection_scenarios():
    transform = InstanceToBoundaryMask(ROLE, connectivity=2)
    
    # Background only (all zeros → all false boundaries)
    labels_bg = torch.zeros(1, 2, 2, 2, dtype=torch.int)
    boundary_bg = transform._instance_to_boundary_mask(labels_bg)
    assert isinstance(boundary_bg, torch.Tensor)
    assert torch.all(~boundary_bg), "Background only should have no boundaries"
    
    # Single voxel instance (boundaries on all faces)
    labels_single = torch.zeros(1, 3, 3, 3, dtype=torch.int)
    labels_single[0, 1, 1, 1] = 1  # Center voxel
    boundary_single = transform._instance_to_boundary_mask(labels_single)
    assert isinstance(boundary_single, torch.Tensor)
    assert boundary_single[0, 1, 1, 1], "Single voxel should be marked as boundary"
    assert torch.sum(boundary_single) >= 6, "Single voxel should have boundaries on all 6 faces"
    
    # Single contiguous instance (boundaries only at edges)
    labels_contiguous = torch.ones(1, 3, 3, 3, dtype=torch.int)
    boundary_contiguous = transform._instance_to_boundary_mask(labels_contiguous)
    assert isinstance(boundary_contiguous, torch.Tensor)
    assert torch.all(~boundary_contiguous), "Interior of contiguous instance should have no boundaries"
    
    # Two separate instances (boundaries around each)
    # Y-Z plane:
    # +-------+
    # | 1 . . |
    # | . . . |
    # | . . 2 |
    # +-------+
    # Expected boundary mask (connectivity=2):
    # +-------+
    # | 1 1 . |
    # | 1 1 1 |
    # | . 1 1 |
    # +-------+

    labels_separate = torch.zeros(1, 3, 3, 3, dtype=torch.int)
    labels_separate[0, 0, 0, :] = 1
    labels_separate[0, 2, 2, :] = 2
    boundary_separate = transform._instance_to_boundary_mask(labels_separate)

    assert isinstance(boundary_separate, torch.Tensor)
    assert torch.all(boundary_separate[0, 0, 0, :]), "Leftward internal boundary should be detected"
    assert torch.all(boundary_separate[0, 1, 1, :]), "External boundary should be detected"
    assert torch.all(boundary_separate[0, 2, 2, :]), "Rightward internal boundary should be detected"
    
    # Two touching instances (boundaries at interface and edges)
    labels_touching = torch.zeros(1, 2, 3, 3, dtype=torch.int)
    labels_touching[0, 0, :, :] = 1
    labels_touching[0, 1, :, :] = 2
    boundary_touching = transform._instance_to_boundary_mask(labels_touching)
    assert isinstance(boundary_touching, torch.Tensor)
    assert torch.any(boundary_touching[0, 0, :, :]), "Touching instance should have boundaries"
    assert torch.any(boundary_touching[0, 1, :, :]), "Touching instance should have boundaries"
    # Boundary should exist at the interface between z=0 and z=1
    assert boundary_touching[0, 0, 1, 1] or boundary_touching[0, 1, 1, 1], "Should have boundary at interface"
    
    # Multiple touching instances
    labels_multi = torch.zeros(1, 2, 2, 2, dtype=torch.int)
    labels_multi[0, 0, 0, 0] = 1
    labels_multi[0, 0, 0, 1] = 2
    labels_multi[0, 0, 1, 0] = 3
    labels_multi[0, 1, 1, 1] = 4
    boundary_multi = transform._instance_to_boundary_mask(labels_multi)
    assert isinstance(boundary_multi, torch.Tensor)
    assert torch.any(boundary_multi), "Multiple instances should have boundaries"


def test_non_batched_tensor_raises_error():
    """InstanceToBoundaryMask should reject unbatched 3D tensors."""
    transform = InstanceToBoundaryMask(ROLE, connectivity=1)
    labels_3d = torch.zeros(2, 2, 2, dtype=torch.int)
    with pytest.raises(ValueError, match="labels must be a 4D tensor assumed to be \\(B,D,H,W\\)"):
        transform._instance_to_boundary_mask(labels_3d)


@pytest.mark.parametrize("connectivity, expected_boundary", [
    (1, {(0, 0), (0, 1), (1, 0)}),              # 6-neighbourhood: face contacts only
    (2, {(0, 0), (0, 1), (1, 0), (1, 1)}),      # 18: edge diagonal (1,1) now touches the 2
    (3, {(0, 0), (0, 1), (1, 0), (1, 1)}),      # 26: same as 18 with no z extent
])
def test_connectivity_controls_which_neighbours_make_a_boundary(connectivity, expected_boundary):
    """A voxel is a boundary when any neighbour in the chosen neighbourhood carries
    a different label: the in-plane diagonal counts only for connectivity >= 2."""
    labels = torch.ones(1, 1, 3, 3, dtype=torch.int)   # no background: only the 1|2 contact counts
    labels[0, 0, 0, 0] = 2
    boundary = InstanceToBoundaryMask(ROLE, connectivity=connectivity)._instance_to_boundary_mask(labels)
    assert {tuple(p) for p in boundary[0, 0].nonzero().tolist()} == expected_boundary


def test_edge_cases():
    transform = InstanceToBoundaryMask(ROLE, connectivity=1)
    
    # Boundaries at image edges (z=0, y=0, x=0 and opposite edges)
    labels_edges = torch.zeros(1, 3, 3, 3, dtype=torch.int)
    labels_edges[0, 1, 1, 1] = 1  # Center voxel
    boundary_edges = transform._instance_to_boundary_mask(labels_edges)
    assert isinstance(boundary_edges, torch.Tensor)
    # Center should have boundaries (it's isolated)
    assert boundary_edges[0, 1, 1, 1], "Center voxel should have boundaries"
    # Edges of the image should be handled correctly (no index errors)
    assert boundary_edges.shape == labels_edges.shape, "Boundary shape should match input"  # type: ignore[union-attr]
    
    # Test instance at z=0 edge
    labels_z0 = torch.zeros(1, 2, 2, 2, dtype=torch.int)
    labels_z0[0, 0, :, :] = 1
    boundary_z0 = transform._instance_to_boundary_mask(labels_z0)
    assert isinstance(boundary_z0, torch.Tensor)
    assert torch.any(boundary_z0[0, 0, :, :]), "Instance at z=0 should have boundaries"
    
    # Test instance at opposite edge
    labels_z1 = torch.zeros(1, 2, 2, 2, dtype=torch.int)
    labels_z1[0, 1, :, :] = 1
    boundary_z1 = transform._instance_to_boundary_mask(labels_z1)
    assert isinstance(boundary_z1, torch.Tensor)
    assert torch.any(boundary_z1[0, 1, :, :]), "Instance at opposite edge should have boundaries"
    
    
    # Known value test: two adjacent voxels with expected boundary positions
    labels_known = torch.zeros(1, 2, 2, 2, dtype=torch.int)
    labels_known[0, 0, 0, 0] = 1
    labels_known[0, 0, 0, 1] = 2  # Adjacent in x-direction
    boundary_known = transform._instance_to_boundary_mask(labels_known)
    assert isinstance(boundary_known, torch.Tensor)
    # Both voxels should be marked as boundaries (they're on the boundary between instances)
    assert boundary_known[0, 0, 0, 0], "First voxel should be on boundary"
    assert boundary_known[0, 0, 0, 1], "Second voxel should be on boundary"
    # The boundary between them should be detected
    assert torch.any(boundary_known), "Boundary should be detected between adjacent voxels"



# ---------------------------------------------------------------------------
# Form-D role-dict contract (what the preprocessor actually feeds these)
# ---------------------------------------------------------------------------


def test_publishes_boundary_role():
    """The transform publishes the derived batched map under its own Form-D role
    -- it does not return a tensor."""
    labels = torch.zeros(1, 3, 3, 3, dtype=torch.int)
    labels[0, 1, 1, 1] = 1

    out = InstanceToBoundaryMask(ROLE, connectivity=1)(_sample(labels))

    tgt = out["metainfo"]["targets"]
    assert list(tgt) == [ROLE, "boundary"], "boundary role published after the source"
    assert tgt["boundary"].shape == labels.shape, "batched (B, D, H, W) derived map"
    assert _derived(out, "boundary")[0, 1, 1, 1], "isolated voxel is a boundary"


def test_batch_elements_are_derived_independently():
    labels = torch.zeros(2, 3, 3, 3, dtype=torch.int)
    labels[0, 1, 1, 1] = 1  # sample 0: one isolated voxel; sample 1 stays empty

    out = InstanceToBoundaryMask(ROLE, connectivity=1)(_sample(labels))
    boundary = _derived(out, "boundary")

    assert boundary[0].any(), "sample 0 has a boundary"
    assert not boundary[1].any(), "sample 1 is empty and must stay empty"


def test_unknown_source_role_raises():
    labels = torch.zeros(1, 2, 2, 2, dtype=torch.int)
    with pytest.raises(KeyError, match="no source role"):
        InstanceToBoundaryMask("not_a_role")(_sample(labels))


def test_foreground_masks_consumes_the_published_boundary():
    """ForegroundMasks(remove_boundary=True) reads the "boundary" role that
    InstanceToBoundaryMask published, so the two are order-dependent."""
    labels = torch.zeros(1, 3, 3, 3, dtype=torch.int)
    labels[0, 1, 1, 1] = 1

    data = InstanceToBoundaryMask(ROLE, connectivity=1)(_sample(labels))
    out = ForegroundMasks(ROLE, remove_boundary=True)(data)

    tgt = out["metainfo"]["targets"]
    assert list(tgt) == [ROLE, "boundary", "foreground"]
    assert tgt["foreground"].shape == labels.shape


def test_foreground_uses_logical_not_on_int_boundary_map():
    """An integer-typed boundary map (e.g. loaded from storage) is negated
    logically, never bitwise: foreground = (label != 0) & (boundary == 0)."""
    fm = ForegroundMasks(source_role="instance", remove_boundary=True)
    labels = torch.tensor([[[[0, 1], [2, 3]]]], dtype=torch.int32)   # (1,1,2,2)
    boundary = torch.tensor([[[[0, 1], [0, 1]]]], dtype=torch.int32)  # int, not bool
    out = fm._foreground_masks(labels, boundary)
    expected = (labels != 0) & (boundary == 0)
    assert torch.equal(out, expected)


def test_foreground_without_boundary_removal_is_every_nonzero_label():
    """remove_boundary=False needs no published "boundary" role: foreground is
    simply every non-background voxel."""
    labels = torch.tensor([[[[0, 1], [2, 0]]]], dtype=torch.int32)
    out = ForegroundMasks(source_role=ROLE, remove_boundary=False)(_sample(labels))
    fg = _derived(out, "foreground")
    assert fg.dtype == torch.bool
    assert torch.equal(fg, labels != 0)
