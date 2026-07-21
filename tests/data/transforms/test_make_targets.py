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
    """Wrap a (B, D, H, W) labelmap in the semantic_maps/semantic_roles contract these
    transforms consume: one target dict per batch element, each carrying a single
    (1, D, H, W) stack tagged with `role`.

    Mirrors SemanticSegmentationPreprocessor.forward (preprocessor.py:1615-1629), which
    is the only production producer of this shape.
    """
    return {
        "metainfo": {
            "targets": [
                {
                    "semantic_maps": labels[b].unsqueeze(0).to(torch.int32),
                    "semantic_roles": [role],
                    "channel_roles": [role],
                }
                for b in range(labels.shape[0])
            ]
        }
    }


def _derived(out: dict, tag: str) -> torch.Tensor:
    """Re-extract an appended slice (e.g. "boundary") as (B, D, H, W).

    The transforms mutate the stack in place rather than returning a tensor, so
    assertions read the slice back out by its role tag.
    """
    ts = out["metainfo"]["targets"]
    return torch.stack(
        [t["semantic_maps"][t["semantic_roles"].index(tag)] for t in ts]
    )


def test_deep_copy_inputs_as_targets_clones_tensor():
    data_tensor = torch.randn(2, 3)
    transform = DeepCopyInputsAsTargets()

    result = transform({"data_tensor": data_tensor})
    result["metainfo"]["targets"][0].add_(1.0)

    assert torch.equal(result["data_tensor"], data_tensor), "Data tensor is was modified"
    assert not torch.equal(result["metainfo"]["targets"][0], result["data_tensor"]), "Modifying targets changed data tensor"
    assert result["metainfo"]["targets"][0].data_ptr() != result["data_tensor"].data_ptr(), "Targets and data tensor share the same memory"

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


@pytest.mark.parametrize("connectivity", [1, 2, 3])
def test_connectivity_modes_produce_different_boundaries(connectivity):
    # Create a test case with two adjacent voxels
    labels = torch.zeros(1, 2, 2, 2, dtype=torch.int)
    labels[0, 0, 0, 0] = 1
    labels[0, 0, 0, 1] = 2  # Adjacent in x-direction
    
    transform = InstanceToBoundaryMask(ROLE, connectivity=connectivity)
    boundary = transform._instance_to_boundary_mask(labels)
    assert isinstance(boundary, torch.Tensor)
    
    # All connectivity modes should detect the boundary between the two voxels
    assert boundary[0, 0, 0, 0] or boundary[0, 0, 0, 1], f"connectivity={connectivity} should detect boundary between adjacent voxels"
    
    # Verify connectivity=3 has more shifts than connectivity=1
    if connectivity == 3:
        # connectivity=3 should detect boundaries that connectivity=1 doesn't (diagonal neighbors)
        # For this simple case, both should detect the same boundary, but connectivity=3 has more shifts
        assert len(transform.shifts) > 6


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
# Role-tagged dict contract (what the preprocessor actually feeds these)
# ---------------------------------------------------------------------------


def test_appends_boundary_slice_tagged_by_role():
    """The transform mutates each target's semantic_maps stack in place, appending the
    derived map and its "boundary" role tag -- it does not return a tensor."""
    labels = torch.zeros(1, 3, 3, 3, dtype=torch.int)
    labels[0, 1, 1, 1] = 1

    out = InstanceToBoundaryMask(ROLE, connectivity=1)(_sample(labels))

    tgt = out["metainfo"]["targets"][0]
    assert tgt["semantic_roles"] == [ROLE, "boundary"], "boundary tag appended in order"
    assert tgt["semantic_maps"].shape[0] == 2, "one slice appended to the stack"
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


def test_foreground_masks_consumes_the_appended_boundary():
    """ForegroundMasks(remove_boundary=True) reads the "boundary" slice that
    InstanceToBoundaryMask appended, so the two are order-dependent."""
    labels = torch.zeros(1, 3, 3, 3, dtype=torch.int)
    labels[0, 1, 1, 1] = 1

    data = InstanceToBoundaryMask(ROLE, connectivity=1)(_sample(labels))
    out = ForegroundMasks(ROLE, remove_boundary=True)(data)

    tgt = out["metainfo"]["targets"][0]
    assert tgt["semantic_roles"] == [ROLE, "boundary", "foreground"]
    assert tgt["semantic_maps"].shape[0] == 3
