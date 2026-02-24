from typing import Any, Optional, Mapping
import inspect
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from omegaconf import DictConfig
from cell_observatory_platform.models.layers.norm import LayerNorm3D
from cell_observatory_platform.models.layers.patch_embeddings import calc_num_patches
from cell_observatory_platform.models.layers.conv3d import Conv3d

try:
    from ops3d import _C
    OPS3D_AVAILABLE = True
except ImportError:
    OPS3D_AVAILABLE = False
    
def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )

class SimpleFeaturePyramid(nn.Module):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    
    Adapted from:
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        patch_shape: tuple[int, int, int],
        input_format: str,
        backbone_embed_dim: int,
        out_channels: int,
        scale_factors: list[float],
        top_block: Optional[nn.Module] = None,
        norm: str = "LayerNorm",
        square_pad: int = 0,
    ):
        """
        Args:
            input_shape (tuple[int, int, int]): shape of the input data.
            patch_shape (tuple[int, int, int]): shape of the patches.
            input_format (str): format of the input data.
            backbone_embed_dim (int): number of channels in the backbone feature map.
            out_channels (int): number of channels in the output feature map.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra pyramid levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            norm (str): the normalization to use.
            square_pad (int): If > 0, require input images to be padded to specific square size.
        """
        super(SimpleFeaturePyramid, self).__init__()
        

        if norm != "LayerNorm":
            raise NotImplementedError(f"Normalization layer {norm} not supported yet.")
        
        self.scale_factors = scale_factors
        strides = [int(16 / scale) for i, scale in enumerate(scale_factors)]
        _assert_strides_are_log2_contiguous(strides)

        self.input_shape = input_shape
        self.patch_shape = patch_shape
        self.input_format = input_format
        # Save shape information for depatchifying
        _, token_shape = calc_num_patches(
            input_fmt=self.input_format,
            input_shape=self.input_shape,
            patch_shape=self.patch_shape,
        )
        if self.input_format == "ZYXC":
            t, z, y, x, c = token_shape
            self.token_shape = [z, y, x]
        else:
            raise NotImplementedError(f"Input format {self.input_format} not supported yet.")
        
        dim = backbone_embed_dim
        self.stages = []
        use_bias = norm == ""
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose3d(dim, dim // 2, kernel_size=2, stride=2),
                    LayerNorm3D(dim // 2), # do 3D layer norm (per channel normalization)
                    # get_norm(norm, dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose3d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool3d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    Conv3d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                        norm=LayerNorm3D(out_channels),
                        # norm=get_norm(norm, out_channels),
                    ),
                    Conv3d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=LayerNorm3D(out_channels),
                        # norm=get_norm(norm, out_channels),
                    ),
                ] # type: ignore
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

        self.top_block = top_block
        # Return feature names are "<stage>", like ["0", "1", "2", "3"]
        self._out_feature_strides = {f"{stage}": stride for stage, stride in enumerate(strides)}
        # top block output feature maps.
        if self.top_block is not None:
            raise NotImplementedError("Top block is not supported yet.")
            # for s in range(stage, stage + self.top_block.num_levels):
            #     self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad

    def _unpatchify(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: list of either [B, N, C] or [B, C, D, H, W]
        if feats.dim() == 3:  # [B, N, C]
            B, N, C = feats.shape
            return feats.transpose(1, 2).reshape(B, C, *self.token_shape)
        else:
            return feats

    @property
    def padding_constraints(self):
        return {
            "size_divisiblity": self._size_divisibility,
            "square_size": self._square_pad,
        }

    def forward_features(self, features: torch.Tensor):
        """
        Args:
            features: Tensor of shape (N,C,Z,Y,X). Z, Y, X must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow 
                the DETR stage naming convention e.g.,
                ["0", "1", "2", "3"].
        """
        results = []
        for stage in self.stages:
            results.append(stage(features))

        # if self.top_block is not None:
        #     if self.top_block.in_feature in bottom_up_features:
        #         top_block_in_feature = bottom_up_features[self.top_block.in_feature]
        #     else:
        #         top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
        #     results.extend(self.top_block(top_block_in_feature))
        
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}

    def forward(self, data_tensor: torch.Tensor, backbone_features: torch.Tensor):
        if isinstance(backbone_features, (list, tuple)):
            backbone_features = backbone_features[-1]
        features = self._unpatchify(backbone_features)
        return self.forward_features(features)

def _extract_model_kwargs(cfg: Mapping[str, Any]) -> dict:
    sig = inspect.signature(SimpleFeaturePyramid.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    ignore = {"_target_", "BUILD"}

    kwargs = {}
    for k, v in cfg.items():
        if k in ignore or k not in allowed:
            continue
        kwargs[k] = v
    return kwargs

def BUILD(adapter_args: dict):
    return SimpleFeaturePyramid(**_extract_model_kwargs(cfg=adapter_args))