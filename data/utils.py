from typing import List, Tuple

import torch
import torch.nn.functional as F


def fft(x: torch.Tensor, axes: tuple[int, ...], pad_to=None):
    if pad_to is not None:
        if isinstance(pad_to, (int, float)):
            target = (int(pad_to),) * len(axes)
        else:
            target = tuple(int(s) for s in pad_to)
            if len(target) != len(axes):
                raise ValueError("len(pad_to) must match number of FFT axes.")

        # center-crop
        for ax, tgt in sorted(zip(axes, target), key=lambda t: t[0]):
            cur = x.shape[ax]
            if tgt < cur:
                start = (cur - tgt) // 2
                stop = start + tgt
                slicer = [slice(None)] * x.ndim
                slicer[ax] = slice(start, stop)
                x = x[tuple(slicer)]

        # zero-pad
        pads = [0, 0] * x.ndim
        for ax, tgt in zip(axes, target):
            cur = x.shape[ax]
            if tgt > cur:
                add = tgt - cur
                before = add // 2
                after = add - before
                i = (x.ndim - 1 - ax) * 2
                pads[i] = before
                pads[i + 1] = after
        if any(pads):
            x = torch.nn.functional.pad(x, pads, mode='constant', value=0)

    x = torch.fft.ifftshift(x, dim=axes)
    X = torch.fft.fftn(x, dim=axes)
    X = torch.fft.fftshift(X, dim=axes)
    return X


# from https://github.com/cell-observatory/aovift/src/synthetic.py
@torch.no_grad()
def create_na_masks(ipsf: torch.Tensor, 
                    thresholds: List[float],
                    target_shape: Tuple[int, int, int] | None,
                    resize: bool = True
):
    assert ipsf.ndim in (2, 3), "PSF must be 2D or 3D"
    ipsf = torch.as_tensor(ipsf, dtype=torch.float32)
    otf = torch.abs(fft(ipsf, axes=tuple(range(ipsf.ndim))))

    # NaN safe max operator
    max_val = torch.where(torch.isnan(otf), otf.new_full((), float("-inf")), otf).max()
    if not torch.isfinite(max_val):
        raise ValueError("OTF is all-NaN — cannot build NA mask")

    # max normalize
    mask = otf / max_val
    
    masks = []
    for thr in thresholds:
        if not (0.0 <= thr <= 1.0):
            raise ValueError(f"Threshold {thr} outside [0,1]")
        # keep magnitudes >= threshold
        binary_mask = (mask >= thr).float()
        if resize and target_shape is not None:
            binary_mask = _resize_mask(binary_mask, target_shape)
        masks.append(binary_mask)
        
    return torch.stack(masks)


def resize_mask(na_mask: torch.Tensor,
                input_format: str,
                channels: int | None = None,
                timepoints: int | None = None,
                axial_shape: int | None = None,
                lateral_shape: tuple[int, int] = None,
                dtype: torch.dtype = None,
                device: torch.device = None):
    if axial_shape is not None and na_mask.shape != (axial_shape, *lateral_shape,):
        mask = _resize_mask(na_mask, (axial_shape, *lateral_shape))
    elif axial_shape is None and na_mask.shape != (*lateral_shape,):
        mask = _resize_mask(na_mask, (*lateral_shape,))
    else:
        mask = na_mask

    view_shape = [1] * len(input_format)
    expand_shape = [1] * len(input_format)

    has_z = 'Z' in input_format and axial_shape is not None and mask.ndim == 3
    y_pos = input_format.index('Y')
    x_pos = input_format.index('X')

    if has_z:
        z_pos = input_format.index('Z')
        if not (z_pos < y_pos < x_pos):
            raise ValueError("Expected spatial order Z<Y<X in input_format for minimal resize/broadcast.")
        z, y, x = mask.shape
        view_shape[z_pos] = z
        view_shape[y_pos] = y
        view_shape[x_pos] = x
    else:
        if not (y_pos < x_pos):
            raise ValueError("Expected spatial order Y<X in input_format for minimal resize/broadcast.")
        y, x = mask.shape
        view_shape[y_pos] = y
        view_shape[x_pos] = x

    # expand: start from spatial view, then set T/C sizes if provided
    expand_shape[:] = view_shape
    if 'T' in input_format and timepoints is not None:
        expand_shape[input_format.index('T')] = timepoints
    if 'C' in input_format and channels is not None:
        expand_shape[input_format.index('C')] = channels

    if dtype is not None or device is not None:
        mask = mask.to(dtype=dtype if dtype is not None else mask.dtype,
                       device=device if device is not None else mask.device)

    mask = mask.view(*view_shape).expand(*expand_shape)

    return mask


def _resize_mask(
    mask: torch.Tensor,
    target_shape: Tuple[int, int, int],
):
    if mask.shape == target_shape:
        return mask

    # input dimensions are interpreted in the form: 
    # mini-batch x channels x [optional depth] x [optional height] x width
    # hence we need to add two singleton dimensions
    if mask.ndim == 3:
        resized = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
    elif mask.ndim == 2:
        resized = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=target_shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
    else:
        raise ValueError("resize_mask only supports 2D or 3D masks")

    return (resized > 0.5).float()


@torch.no_grad()
def downsample(
    na_mask: torch.Tensor, 
    inputs: torch.Tensor, 
    spatial_dims: tuple[int, ...],
    batched_computation: bool = True,
):  
    if not batched_computation:    
        if inputs.dtype == torch.bfloat16:
            inputs = inputs.to(dtype=torch.float32)
        # FFT, shift to centre
        k = torch.fft.fftn(inputs, dim=spatial_dims)
        k = torch.fft.fftshift(k,  dim=spatial_dims)

        # clip: element-wise multiply 
        na_mask = na_mask.to(dtype=k.real.dtype, device=k.device)
        k.mul_(na_mask)

        # shift back and inverse FFT
        k = torch.fft.ifftshift(k, dim=spatial_dims)
        out = torch.fft.ifftn(k, dim=spatial_dims).real
        
        if inputs.dtype == torch.bfloat16:
            out = out.to(dtype=torch.bfloat16)
        
        return out
    
    out = torch.empty_like(inputs)
    B = inputs.shape[0]
    for b in range(B):
        x_b = inputs.narrow(0, b, 1)
        xb = x_b.to(torch.float32)
        k = torch.fft.fftn(xb, dim=spatial_dims)
        k = torch.fft.fftshift(k, dim=spatial_dims)
        k.mul_(na_mask)
        k = torch.fft.ifftshift(k, dim=spatial_dims)
        yb = torch.fft.ifftn(k, dim=spatial_dims).real
        out.narrow(0, b, 1).copy_(yb.to(inputs.dtype))
    
    return out