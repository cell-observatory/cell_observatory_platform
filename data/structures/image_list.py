"""
Adopted with Apache License 2.0 from
https://github.com/facebookresearch/detectron2/blob/main/detectron2/structures/image_list.py
"""

from typing import Any, List, Tuple, Optional, Dict

import torch
from torch import device

from data.io import record_init
from data.data_shapes import MULTICHANNEL_3D_HYPERCUBE, MULTICHANNEL_4D_HYPERCUBE


class ImageList:
    """
        Structure that holds a list of images (of possibly varying sizes)
        as a single tensor. This works by padding the images to the same size.
        The original size of each image is stored in `image_sizes`.
    """

    @record_init
    def __init__(
        self,
        tensor: torch.Tensor,
        image_sizes: List[Tuple],
        layout: MULTICHANNEL_3D_HYPERCUBE | MULTICHANNEL_4D_HYPERCUBE = MULTICHANNEL_4D_HYPERCUBE.TZYXC,
        orig_layout: MULTICHANNEL_3D_HYPERCUBE | MULTICHANNEL_4D_HYPERCUBE = None,
    ):
        """
        Arguments:
            tensor (Tensor): of shape (N, [T,D], H, W) or (N, C_1, ..., C_K, [T,D], H, W) where K >= 1
            image_sizes (List[Tuple[int, int, int]] | List[Tuple[int, int, int, int]])
            layout (MULTICHANNEL_HYPERCUBE): Desired tensor layout
            orig_layout (MULTICHANNEL_HYPERCUBE): Current tensor layout
        """
        self.tensor = tensor
        self.image_sizes = image_sizes

        self.layout = layout
        self.orig_layout = orig_layout if orig_layout is not None else layout

        if self.layout != self.orig_layout:
            if self.layout == MULTICHANNEL_3D_HYPERCUBE.TYXC or self.layout == MULTICHANNEL_4D_HYPERCUBE.TZYXC:
                self.tensor = self.layout.to_channel_last(self.tensor)
            elif self.layout == MULTICHANNEL_3D_HYPERCUBE.CTYX or self.layout == MULTICHANNEL_4D_HYPERCUBE.CTZYX:
                self.tensor = self.layout.to_channel_first(self.tensor)
            else:
                raise NotImplementedError(f"Unsupported layout {self.layout}")

        if isinstance(self.layout, MULTICHANNEL_3D_HYPERCUBE):
            if self.tensor.ndim == 4:
                self.tensor = self.tensor.unsqueeze(0) # (C, D, H, W) -> (1, C, D, H, W)

        elif isinstance(self.layout, MULTICHANNEL_4D_HYPERCUBE):
            if self.tensor.ndim == 5:
                self.tensor = self.tensor.unsqueeze(0) # (T, C, D, H, W) -> (1, T, C, D, H, W)

    @property
    def has_time(self) -> bool:
        return self.layout.has_temporal_dim()

    @property
    def num_timepoints(self) -> int | None:
        return self.layout.num_timepoints(self.tensor)

    @property
    def num_channels(self) -> int:
        return self.layout.num_channels(self.tensor)

    @property
    def image_shape(self) -> Tuple[int, int, int] | Tuple[int, int]:
        return self.layout.get_spatial_shape(self.tensor)

    @property
    def shape(self) -> Tuple:
        return self.tensor.shape

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any) -> "ImageList":
        cast_tensor = self.tensor.to(*args, **kwargs)
        return ImageList(cast_tensor, image_sizes=self.image_sizes, layout=self.layout, orig_layout=self.orig_layout)

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    @property
    def device(self) -> device:
        return self.tensor.device

    def __len__(self) -> int:
        return len(self.image_sizes)

    def __getitem__(self, idx) -> torch.Tensor:
        """ Access the individual image in its original size. """
        s = self.image_sizes[idx]

        if self.layout == MULTICHANNEL_3D_HYPERCUBE:
            if self.layout.is_channel_first():
                return self.tensor[idx, :, :s[0], :s[1], :s[2]]
            else:
                return self.tensor[idx, :s[0], :s[1], :s[2]]

        elif self.layout == MULTICHANNEL_4D_HYPERCUBE:
            if self.layout.is_channel_first():
                return self.tensor[idx, :, :s[0], :s[1], :s[2], :s[3]]
            else:
                return self.tensor[idx, :s[0], :s[1], :s[2], :s[3]]

        else:
            raise NotImplementedError(f"Unknown layout {self.layout}")

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __repr__(self) -> str:
        shape = self.layout.get_image_shape_dict(self.tensor)

        return (
            f"<ImageList  "
            f"N={self.tensor.shape[0]} | {shape} "
            f"layout={self.layout}  "
            f"orig_layout={self.orig_layout}  "
            f"device={self.tensor.device}>"
        )


    def copy(self, *, deep: bool = False) -> "ImageList":
        return ImageList(
            self.tensor.clone() if deep else self.tensor,
            self.image_sizes.copy(),
            layout=self.layout,
            orig_layout=self.orig_layout,
        )

    def get_image_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.layout == MULTICHANNEL_3D_HYPERCUBE:
            if self.layout.is_channel_first(): # (B, C, D, H, W)
                mean = self.tensor.mean(dim=(2, 3, 4), keepdim=True)
                std = self.tensor.std(dim=(2, 3, 4), keepdim=True)
            else:  # (B, D, H, W, C)
                mean = self.tensor.mean(dim=(1, 2, 3), keepdim=True)
                std = self.tensor.std(dim=(1, 2, 3), keepdim=True)

        elif self.layout == MULTICHANNEL_4D_HYPERCUBE:
            if self.layout.is_channel_first(): # (B, C, T, D, H, W)
                mean = self.tensor.mean(dim=(2, 3, 4, 5), keepdim=True)
                std = self.tensor.std(dim=(2, 3, 4, 5), keepdim=True)
            else: # (B, T, D, H, W, C)
                mean = self.tensor.mean(dim=(1, 2, 3, 4), keepdim=True)
                std = self.tensor.std(dim=(1, 2, 3, 4), keepdim=True)

        else:
            raise NotImplementedError(f"Unsupported layout {self.layout}")

        return mean, std


    @staticmethod
    def from_tensors(
        tensors: List[torch.Tensor],
        layout: MULTICHANNEL_3D_HYPERCUBE | MULTICHANNEL_4D_HYPERCUBE = MULTICHANNEL_4D_HYPERCUBE.TZYXC,
        size_divisibility: int = 0,
        pad_value: float = 0.0,
        padding_constraints: Optional[Dict[str, int]] = None,
    ) -> "ImageList":
        """
        Args:
        tensors: a tuple or list of `torch.Tensor`, Each tuple is ([t, d], h, w, c) or (c, [t, d], h, w).
            The Tensors will be padded to the same shape with `pad_value`.
        size_divisibility (int): If `size_divisibility > 0`, add padding to ensure
            the common height and width is divisible by `size_divisibility`.
            This depends on the model and many models need a divisibility of 32.
        pad_value (float): value to pad.
        padding_constraints (optional[Dict]): If given, it would follow the format as
            {"size_divisibility": int, "square_size": int}, where `size_divisibility` will
            overwrite the above one if presented and `square_size` indicates the
            square padding size if `square_size` > 0.
        Returns:
            an `ImageList`.
        """
        assert len(tensors) > 0

        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert isinstance(t, torch.Tensor), type(t)

        image_sizes = [layout.get_spatial_shape(im) for im in tensors] # List[Tuple[:]]
        image_sizes_tensor = [torch.as_tensor(x) for x in image_sizes]
        max_size = torch.stack(image_sizes_tensor).max(0).values  # List[Nx:] -> (N, :) -> (:,)

        if padding_constraints is not None:
            sq = padding_constraints.get("square_size", 0)

            if sq > 0: # pad to square.
                max_size = (sq for s in max_size)

                if "size_divisibility" in padding_constraints:
                    size_divisibility = padding_constraints["size_divisibility"]

        if size_divisibility > 1:
            stride = size_divisibility
            max_size = (max_size + (stride - 1)).div(stride, rounding_mode="floor") * stride

        batch_size = len(tensors)
        num_channels = layout.num_channels(tensors[0])

        if layout.has_temporal_dim():
            num_timepoints = layout.num_timepoints(tensors[0])
            if layout.is_channel_last():
                output_shape = [batch_size, num_timepoints, *max_size, num_channels]
            else:
                output_shape = [batch_size, num_channels, num_timepoints, *max_size]
        else:
            if layout.is_channel_last():
                output_shape = [batch_size, *max_size, num_channels]
            else:
                output_shape = [batch_size, num_channels, *max_size]

        padded_tensor = tensors[0].new_full(output_shape, fill_value=pad_value)
        padded_tensor = padded_tensor.to(tensors[0].device)

        for i, img in enumerate(tensors): # fill in the tensor with the images
            s = layout.get_spatial_shape(img)
            if len(s) == 2: # 2D image
                if layout.has_temporal_dim():
                    if layout.is_channel_last(): # (batch_size, T, Y, X, C)
                        padded_tensor[i, :, :s[0], :s[1], :].copy_(img)
                    else:  # (batch_size, C, T, Y, X)
                        padded_tensor[i, :, :, :s[0], :s[1]].copy_(img)
                else:
                    if layout.is_channel_last(): # (batch_size, Y, X, C)
                        padded_tensor[i, :s[0], :s[1], :].copy_(img)
                    else: # (batch_size, C, Y, X)
                        padded_tensor[i, :, :s[0], :s[1]].copy_(img)

            else: # 3D volume
                if layout.has_temporal_dim():
                    if layout.is_channel_last(): # (batch_size, T, Z, Y, X, C)
                        padded_tensor[i, :, :s[0], :s[1], :s[2], :].copy_(img)
                    else:  # (batch_size, C, T, Z, Y, X)
                        padded_tensor[i, :, :, :s[0], :s[1], :s[2]].copy_(img)
                else:
                    if layout.is_channel_last(): # (batch_size, Z, Y, X, C)
                        padded_tensor[i, :s[0], :s[1], :s[2], :].copy_(img)
                    else: # (batch_size, C, Z, Y, X)
                        padded_tensor[i, :, :s[0], :s[1], :s[2]].copy_(img)

        return ImageList(padded_tensor.contiguous(), image_sizes=image_sizes, layout=layout)
