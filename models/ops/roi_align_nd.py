"""
https://github.com/TimothyZero/MedVision/blob/main/medvision/ops/roi_align_nd.py

Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn
from torch.autograd import Function

try:
    from ops3d import _C
except ImportError:
    print("RoIAlign3DFunction op failed to load. Please compile ops3d if needed.")


class RoIAlign3DFunction(Function):
    @staticmethod
    def forward(ctx, features, rois, output_size, spatial_scale, sampling_ratio=0, order=1):
        if isinstance(output_size, int):
            out_d = output_size
            out_h = output_size
            out_w = output_size
        elif isinstance(output_size, tuple):
            assert len(output_size) == 3
            assert isinstance(output_size[0], int)
            assert isinstance(output_size[1], int)
            assert isinstance(output_size[2], int)
            out_d, out_h, out_w = output_size
        else:
            raise TypeError('"output_size" must be an integer or tuple of integers')
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.order = order
        ctx.save_for_backward(rois)
        ctx.feature_size = features.size()

        batch_size, num_channels, data_depth, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new_zeros(num_rois, num_channels, out_d, out_h, out_w)
        _C.roi_align_3d_forward(features, rois, output, out_d, out_h, out_w, spatial_scale, sampling_ratio, order)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        feature_size = ctx.feature_size
        spatial_scale = ctx.spatial_scale
        order = ctx.order
        sampling_ratio = ctx.sampling_ratio
        rois = ctx.saved_tensors[0]
        assert feature_size is not None
        batch_size, num_channels, data_depth, data_height, data_width = feature_size

        out_w = grad_output.size(4)
        out_h = grad_output.size(3)
        out_d = grad_output.size(2)

        grad_input = grad_rois = None

        if ctx.needs_input_grad[0]:
            # dtype/device must follow grad_output (the features' grad), not the
            # rois tensor — rois may be a different dtype than the feature map.
            grad_input = grad_output.new_zeros(batch_size, num_channels, data_depth, data_height, data_width)
            _C.roi_align_3d_backward(
                grad_output.contiguous(), rois, grad_input, out_d, out_h, out_w, spatial_scale, sampling_ratio, order
            )
        return grad_input, grad_rois, None, None, None, None, None


# roi_align_2d = RoIAlign2DFunction.apply
roi_align_3d = RoIAlign3DFunction.apply


class RoIAlign(nn.Module):
    """RoI align pooling layer for proposals.

    It accepts a feature map of shape (N, C, H, W) or (N, C, D, H, W) and rois with shape
    (n, 5) or (n, 7) with each roi decoded as (batch_index, x1, y1, x2, y2) or
    (batch_index, x1, y1, z1, x2, y2, z2).

    Args:
        output_size (tuple): (h, w) or (d,h,w)
        spatial_scale (float): scale the input boxes by this number
        sampling_ratio (int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
    Note:
        The implementation of RoIAlign is modified from
        https://github.com/open-mmlab/mmdetection
    """

    def __init__(self, output_size: tuple, spatial_scale: float, sampling_ratio: int = -1):
        super(RoIAlign, self).__init__()
        assert iter(output_size)

        self.output_size = output_size
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.dim = len(self.output_size)

        if self.dim == 3:
            self.roi_align = roi_align_3d
        else:
            raise Exception("Tried to init RoIAlign module with incorrect output size: {}".format(self.output_size))

    def forward(self, features, rois, order=1):
        assert order in [0, 1], "only support order = 0 (nearest) or 1 (linear)"
        sampling_ratio = 1 if order == 0 else self.sampling_ratio
        return self.roi_align(features, rois, self.output_size, self.spatial_scale, sampling_ratio, order)
