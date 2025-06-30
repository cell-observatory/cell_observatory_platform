import torch

from data.structures.image_list import ImageList
from data.structures.data_sample import DataSample
from data.data_shapes import MULTICHANNEL_3D_HYPERCUBE


def test_data_sample():
    # build an ImageList
    layout = MULTICHANNEL_3D_HYPERCUBE.ZYXC
    img1 = torch.randn(2, 4, 4, 1)
    img2 = torch.randn(3, 5, 6, 1)
    im_list = ImageList.from_tensors([img1, img2], layout=layout)

    # build DataSample
    meta = {"id": 123, "note": "unit-test"}
    ds = DataSample(metainfo=meta)
    ds.data_tensor = im_list

    # sanity checks
    assert ds.data_tensor is im_list
    assert ds.metainfo == meta

    serial = ds.to_dict()

    # check required keys present
    assert set(serial) == {"metainfo", "data_tensor", "data_tensor_meta"}

    clone = DataSample.from_dict(serial)

    # metadata preserved
    assert clone.metainfo == meta

    # tensor & layout preserved
    assert torch.allclose(clone.data_tensor.tensor, im_list.tensor)
    assert clone.data_tensor.layout == im_list.layout
    assert clone.data_tensor.image_sizes == im_list.image_sizes