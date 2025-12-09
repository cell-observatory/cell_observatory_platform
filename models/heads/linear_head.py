""" 
Adapted from:
https://github.com/facebookresearch/dinov3/blob/main/dinov3/layers/dino_head.py
"""

import inspect
from typing import Mapping, Any

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


class LinearHead(nn.Module):
    def __init__(
        self,
        in_dim,
        output_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        nlayers = max(nlayers, 1)
        
        self.mlp = _build_mlp(
            nlayers,
            in_dim,
            bottleneck_dim,
            hidden_dim=hidden_dim,
            use_bn=use_bn,
            bias=mlp_bias,
        )
        self.last_layer = nn.Linear(bottleneck_dim, self.output_dim, bias=False)

    def init_weights(self) -> None:
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, no_last_layer=False, only_last_layer=False):
        if not only_last_layer:
            x = self.mlp(x)
            eps = 1e-6 if x.dtype == torch.float16 else 1e-12
            x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        if not no_last_layer:
            x = self.last_layer(x)
        return x


def _build_mlp(nlayers, 
               in_dim, 
               bottleneck_dim, 
               hidden_dim=None, 
               use_bn=False, 
               bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)
    

def _extract_model_kwargs(cfg: Mapping[str, Any]) -> dict:
    cfg = dict(cfg)

    # Mandatory: AutoBench must set input_dim
    in_dim = cfg.get("input_dim", None)
    out_dim = cfg.get("output_dim", None)
    if in_dim is None or out_dim is None:
        raise ValueError("input_dim must be specified in the config for MaskedPredictor")

    # Map generic `input_dim` to the actual args
    cfg["in_dim"] = in_dim
    cfg["output_dim"] = out_dim

    sig = inspect.signature(LinearHead.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    ignore = {"_target_", "BUILD"}

    kwargs = {}
    for k, v in cfg.items():
        if k in ignore or k not in allowed:
            continue
        kwargs[k] = v
    return kwargs


def BUILD(cfg: Mapping[str, Any]) -> LinearHead:
    """
    Hydra entrypoint for LinearHead.

    Accepts both:
      - in_dim / nlayers
      - input_dim / num_layers  (aliased to above)
    """
    return LinearHead(**_extract_model_kwargs(cfg))