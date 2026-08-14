import functools
from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from hydra.utils import get_method

from cell_observatory_platform.models.layers.norm import LayerNorm3D, LayerNorm4D
from cell_observatory_platform.models.layers.patch_embeddings import calc_num_patches
from cell_observatory_platform.models.layers.positional_encoding import PositionalEmbeddingSinCos


class SAMBackbone(nn.Module):
    def __init__(
        self,
        backbone_args: dict,
        adapter_args: Optional[dict],
        backbone_embed_dims: List[int],
        train_backbone: bool,
        blocks_to_train: Optional[List[str]] = None,
        use_layernorm: bool = True,
        adapter_out_layers: Optional[List[int]] = None,
        backbone_output_format: str = "feature_map",
        input_shape: Optional[List[int]] = [1, 128, 256, 512, 2],
        patch_shape: Optional[List[int]] = [1, 16, 32, 32, None],
        input_format: Optional[str] = "TZYXC",
        position_encoding_type: Optional[str] = "sincos",
        pos_encoding_temperature: Optional[int] = 10000,
        pos_encoding_normalize: Optional[bool] = False,
        use_sam_channel_projection: bool = False,
        backbone_native_channels: Optional[int] = None,
    ):
        super().__init__()

        self.input_format = input_format
        self.use_sam_channel_projection = use_sam_channel_projection
        if use_sam_channel_projection:
            assert backbone_native_channels is not None, "backbone_native_channels required when use_sam_channel_projection=True"

        self.backbone = REGISTRY.build("backbone", backbone_args.name, backbone_args)

        self.backbone_embed_dims = backbone_embed_dims

        assert position_encoding_type is not None, "position_encoding_type must be specified"
        self.position_encoding_type = position_encoding_type
        if self.position_encoding_type == "sincos":
            # assert backbone_embed_dims[-1] % 3 == 0, (
            #     f"backbone_embed_dims[-1]={backbone_embed_dims[-1]} must be divisible by 3 "
            #     "for 3D sincos positional encoding (Z/Y/X split)."
            # )
            self.position_encoding = PositionalEmbeddingSinCos(
                # NOTE: split the embedding dim into 3 for each spatial dimension
                num_pos_feats=backbone_embed_dims[-1] // 3,
                temperature=pos_encoding_temperature,
                normalize=pos_encoding_normalize,
                scale=None,
                output_dim=backbone_embed_dims[-1],
            )
        else:
            raise NotImplementedError(f"Position encoding type {self.position_encoding_type} not supported yet.")

        self.blocks_to_train = blocks_to_train

        for _, (name, parameter) in enumerate(self.backbone.named_parameters()):
            train_condition = any(f".{b}." in name for b in self.blocks_to_train) if self.blocks_to_train else True
            if (not train_backbone) or "mask_token" in name or (not train_condition):
                parameter.requires_grad_(False)

        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            if self.input_format == "TZYXC":
                ln_dim = backbone_native_channels if use_sam_channel_projection else None
                self.layer_norms = nn.ModuleList([
                    LayerNorm3D(ln_dim or embed_dim) for embed_dim in backbone_embed_dims
                ])
            else:
                raise NotImplementedError(f"Input format {self.input_format} not supported yet.")

        if use_sam_channel_projection and self.input_format == "TZYXC":
            out_dim = backbone_embed_dims[-1]
            self.sam_channel_projection = nn.Sequential(
                nn.Conv3d(backbone_native_channels, out_dim, kernel_size=1),
                LayerNorm3D(out_dim),
                nn.Conv3d(out_dim, out_dim, kernel_size=3, padding=1),
                LayerNorm3D(out_dim),
            )
        else:
            self.sam_channel_projection = None

        if adapter_args is not None:
            self.with_backbone_adapter = True
            self.adapter = REGISTRY.build("adapter", adapter_args.name, adapter_args)
        else:
            # TODO: implement logic to handle positional encodings without adapter
            self.with_backbone_adapter = False

        self.adapter_out_layers = adapter_out_layers

        if len(backbone_embed_dims) > 1:
            self.multi_scale_features = True
        else:
            self.multi_scale_features = False

        self.input_shape = input_shape
        self.patch_shape = patch_shape
        _, token_shape = calc_num_patches(
            input_fmt=self.input_format,
            input_shape=self.input_shape,
            patch_shape=patch_shape,
        )
        if self.input_format == "TZYXC":
            t, z, y, x, c = token_shape
            # input data is 4D but SAM models flatten B,T 
            self.token_shape = [z, y, x]
        else:
            raise NotImplementedError(f"Input format {self.input_format} not supported yet.")

        assert self.input_format[-1] == "C", "The last dimension of input_format must be 'C'."
        self.out_channels = self.input_shape[-1]
        self.backbone_output_format = backbone_output_format
        self.backbone_returns_sequence = self.backbone_output_format == "sequence"
        # What the backbone CONSUMES (independent of what it returns):
        # PatchEmbedding-based encoders (masked_vit / masked_hiera) patchify
        # channels-last input_format; conv backbones take conv layout. The
        # hiera-multiscale config returns feature_map but still consumes
        # channels-last, so this cannot key off backbone_output_format.
        self.backbone_consumes_channels_last = (
            getattr(self.backbone, "patch_embedding", None) is not None
        )
        
    def _unpatchify_if_sequence(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        # feats: list of either [B, N, C] or [B, C, D, H, W]
        if not self.backbone_returns_sequence:
            return feats

        out = []
        for feat in feats:
            if feat.dim() == 3:  # [B, N, C]
                B, N, C = feat.shape
                out.append(feat.transpose(1, 2).reshape(B, C, *self.token_shape))
            else:
                out.append(feat)
        return out

    def _make_backbone_output(self, feats: List[torch.Tensor]) -> dict:
        """
        Build the standard output dict expected by sam.py:
        {"backbone_fpn": [feat, ...], "vision_pos_enc": [pe, ...]}.
        """
        feats = [f for f in feats if f is not None]
        assert self.input_format == "TZYXC", f"Expected input_format 'TZYXC', got {self.input_format}"
        assert all(f.dim() == 5 for f in feats), (
            f"Expected 5D feature maps [B,C,D,H,W], got {[f.shape for f in feats]}"
        )
        # Sort finest (largest spatial) to coarsest
        feats = sorted(
            feats,
            key=lambda t: t.shape[-3] * t.shape[-2] * t.shape[-1],
            reverse=True,
        )
        position_encodings = [self.position_encoding(f) for f in feats]
        return {
            "backbone_fpn": feats,
            "vision_pos_enc": position_encodings,
        }

    def _to_backbone_layout(self, x: torch.Tensor) -> torch.Tensor:
        """SAM2 hands every backbone conv layout ``(B*T, C, Z, Y, X)``
        (``SAM2._to_model_layout``). PatchEmbedding backbones (masked_vit /
        masked_hiera) patchify **channels-last** ``input_format`` -- with T=1
        the numel matches and the channels-first tensor would reshape into
        scrambled tokens silently (every token mixing unrelated voxels and
        channels). Convert at this boundary; conv backbones keep conv layout.
        """
        if not self.backbone_consumes_channels_last:
            return x
        x = x.permute(0, 2, 3, 4, 1)              # (B*T, C, Z, Y, X) -> (B*T, Z, Y, X, C)
        if self.input_format.startswith("T"):
            x = x.unsqueeze(1)
        return x

    def _to_adapter_layout(self, x: torch.Tensor) -> torch.Tensor:
        """EncoderAdapter.forward assumes channels-last ``(B*T, Z, Y, X, C)`` and
        permutes to conv layout internally. SAM2 hands conv layout
        ``(B*T, C, Z, Y, X)`` (``SAM2._to_model_layout``); convert at this boundary.
        Unlike ``_to_backbone_layout`` there is NO temporal unsqueeze -- the
        adapter's spatial prior module is purely spatial 3D."""
        return x.permute(0, 2, 3, 4, 1).contiguous()  # (B*T, C, Z, Y, X) -> (B*T, Z, Y, X, C)

    def forward(self, data_sample: dict):
        feats = self.backbone.forward_features(
            self._to_backbone_layout(data_sample["data_tensor"])
        )

        adapter_keys = None
        # NOTE: SAM2 uses FPN neck to extract features from multi-scale backbone
        #       if we use simple ViT backbone, opt for VitDET style adapter instead
        if self.with_backbone_adapter:
            feats_dict = self.adapter(
                self._to_adapter_layout(data_sample["data_tensor"]), feats
            )
            feats_dict = {str(k): v for k, v in feats_dict.items()}  # ensure string keys
            adapter_keys = sorted(feats_dict.keys(), key=lambda s: int(s))
            feats_list = [feats_dict[k] for k in adapter_keys]
        else:
            if isinstance(feats, (list, tuple)):
                feats_list = list(feats)
            else:
                feats_list = [feats]

        feats_list = self._unpatchify_if_sequence(feats_list)
        if self.use_layernorm:
            assert len(self.layer_norms) == len(feats_list), (
                f"layer_norms ({len(self.layer_norms)}) != feats_list ({len(feats_list)})"
            )
            feats_list = [ln(f).contiguous() for ln, f in zip(self.layer_norms, feats_list)]

        if self.adapter_out_layers is not None:
            feats_list = [feats_list[i] for i in self.adapter_out_layers]

        if self.sam_channel_projection is not None:
            feats_list = [self.sam_channel_projection(f).contiguous() for f in feats_list]

        return self._make_backbone_output(feats_list)


from cell_observatory_platform.utils.registry import REGISTRY


@REGISTRY.register("backbone", "sam")
def BUILD(backbone_wrapper_args: dict, adapter_args: Optional[dict] = None) -> nn.Module:
    out_layers = backbone_wrapper_args.get("out_layers", None)
    if out_layers is not None:
        backbone_wrapper_args["backbone_args"]["out_layers"] = out_layers

    model = SAMBackbone(
        backbone_args=backbone_wrapper_args["backbone_args"],
        adapter_args=adapter_args,
        backbone_embed_dims=backbone_wrapper_args["backbone_embed_dims"],
        train_backbone=backbone_wrapper_args["train_backbone"],
        blocks_to_train=backbone_wrapper_args.get("blocks_to_train"),
        use_layernorm=backbone_wrapper_args.get("use_layernorm", True),
        adapter_out_layers=backbone_wrapper_args.get("adapter_out_layers"),
        backbone_output_format=backbone_wrapper_args.get("backbone_output_format"),
        input_shape=backbone_wrapper_args.get("input_shape"),
        input_format=backbone_wrapper_args.get("input_format"),
        patch_shape=backbone_wrapper_args.get("patch_shape"),
        position_encoding_type=backbone_wrapper_args.get("position_encoding_type"),
        use_sam_channel_projection=backbone_wrapper_args.get("use_sam_channel_projection", False),
        backbone_native_channels=backbone_wrapper_args.get("backbone_native_channels"),
    )
    return model