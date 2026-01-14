import pytest
import torch

from cell_observatory_platform.data.transforms.make_targets import DeepCopyInputsAsTargets


def test_deep_copy_inputs_as_targets_clones_tensor():
    data_tensor = torch.randn(2, 3)
    transform = DeepCopyInputsAsTargets()

    result = transform({"data_tensor": data_tensor})
    result["metainfo"]["targets"][0].add_(1.0)

    assert torch.equal(result["data_tensor"], data_tensor), "Data tensor is was modified"
    assert not torch.equal(result["metainfo"]["targets"][0], result["data_tensor"]), "Modifying targets changed data tensor"
    assert result["metainfo"]["targets"][0].data_ptr() != result["data_tensor"].data_ptr(), "Targets and data tensor share the same memory"


def test_deep_copy_inputs_as_targets_rejects_existing_targets():
    transform = DeepCopyInputsAsTargets()
    data = {"data_tensor": torch.zeros(1), "metainfo": {"targets": [torch.ones(1)]}}

    with pytest.raises(ValueError):
        transform(data)


def test_deep_copy_inputs_as_targets_requires_data_tensor():
    transform = DeepCopyInputsAsTargets()

    with pytest.raises(KeyError):
        transform({})

