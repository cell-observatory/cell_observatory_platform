import pytest
import torch

from cell_observatory_platform.data.class_structures.image_list import ImageList, cat_image_lists
from cell_observatory_platform.data.data_shapes import MULTICHANNEL_HYPERCUBE

torch.manual_seed(0)


def _rand(shape):
    return torch.randn(*shape)


def _assert_stats_correct(tensor, mean, std, dims):
    m = tensor.mean(dim=dims, keepdim=True)
    s = tensor.std(dim=dims, keepdim=True)
    assert torch.allclose(m, mean, atol=1e-6)
    assert torch.allclose(s, std, atol=1e-6)


# ---------------------------------------------------------------------------
# 3-D CHANNEL-LAST  (ZYXC)
# ---------------------------------------------------------------------------


def test_image_list_3d_channel_last():
    layout = MULTICHANNEL_HYPERCUBE.ZYXC
    # img0:   (Z=2, Y=4, X=4, C=1)
    # img1:   (Z=3, Y=5, X=6, C=1)
    imgs = [_rand((2, 4, 4, 1)), _rand((3, 5, 6, 1))]

    ilist = ImageList.from_tensors(imgs, layout=layout, pad_value=0.0)
    assert len(ilist) == 2
    # padded shape should be (N=2, Z=3, Y=5, X=6, C=1)
    assert ilist.tensor.shape == (2, 3, 5, 6, 1)
    assert ilist.image_sizes == [(2, 4, 4), (3, 5, 6)]
    assert ilist.has_time is False
    assert ilist.num_channels == 1
    assert ilist.image_shape == (3, 5, 6)

    # __getitem__ returns original sizes
    assert ilist[0].shape == imgs[0].shape
    assert ilist[1].shape == imgs[1].shape

    mean, std = ilist.get_image_stats()
    _assert_stats_correct(ilist.tensor, mean, std, dims=(1, 2, 3))

    # copy() deep
    clone = ilist.copy(deep=True)
    clone.tensor.add_(1.0)
    assert not torch.allclose(clone.tensor, ilist.tensor)

    # __repr__
    assert "N=2" in repr(ilist)


# ---------------------------------------------------------------------------
# 4-D CHANNEL-LAST (TZYXC)
# ---------------------------------------------------------------------------


def test_image_list_4d_channel_last():
    layout = MULTICHANNEL_HYPERCUBE.TZYXC
    # img0: (T=3, Z=2, Y=4, X=4, C=1)
    # img1: (T=3, Z=3, Y=5, X=6, C=1)
    imgs = [_rand((3, 2, 4, 4, 1)), _rand((3, 3, 5, 6, 1))]
    ilist = ImageList.from_tensors(imgs, layout=layout, pad_value=0.0)

    # shape  (N=2, T=3, Z=3, Y=5, X=6, C=1)
    assert ilist.tensor.shape == (2, 3, 3, 5, 6, 1)
    assert ilist.has_time is True
    assert ilist.num_timepoints == 3
    assert ilist.num_channels == 1
    assert ilist.image_shape == (3, 5, 6)

    # __getitem__
    assert ilist[0].shape == imgs[0].shape
    assert ilist[1].shape == imgs[1].shape

    # stats
    mean, std = ilist.get_image_stats()
    _assert_stats_correct(ilist.tensor, mean, std, dims=(1, 2, 3, 4))


# ---------------------------------------------------------------------------
# cat_image_lists
# ---------------------------------------------------------------------------


def test_cat_image_lists():
    layout = MULTICHANNEL_HYPERCUBE.ZYXC
    a = ImageList.from_tensors([_rand((2, 4, 4, 1))], layout=layout)
    b = ImageList.from_tensors([_rand((2, 4, 4, 1))], layout=layout)
    c = ImageList.from_tensors([_rand((3, 5, 6, 1))], layout=layout)

    # same shape branch
    ab = cat_image_lists([a, b])
    assert ab.tensor.shape[0] == 2
    assert len(ab.image_sizes) == 2

    # different shapes branch (forces re-padding)
    ac = cat_image_lists([a, c])
    # padded to larger size
    assert ac.tensor.shape[-4:-1] == (3, 5, 6)
    assert len(ac) == 2
