"""
https://github.com/TimothyZero/MedVision/blob/main/medvision/ops/nms_nd.py

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

import torch

try:
    from ops3d import _C
except ImportError:
    print("nms_nd op failed to load. Please compile ops3d if needed.")


def nms_nd(dets: torch.Tensor, iou_threshold: float):
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than iou_threshold with another (higher scoring)
    box.

    Args:
        dets : Tensor[N, 4+1] for 2D or Tensor[N, 6+1] for 3D.
            boxes to perform NMS on. They
            are expected to be in (x1, y1, x2, y2, score) or
            (x1, y1, z1, x2, y2, z2, score) format
        iou_threshold : float
            discards all overlapping
            boxes with IoU < iou_threshold

    Returns:
    keep : Tensor
        int64 tensor with the indices
        of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    # dim = dets.shape[-1] // 2
    boxes = dets[:, :-1]
    scores = dets[:, -1]
    return _C.nms_3d(boxes, scores, iou_threshold), dets
