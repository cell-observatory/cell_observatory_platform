# from typing import Optional, Tuple, Union, Any, Dict

# import torch


# class ChannelDropout:
#     """
#     Randomly drop channels by selecting a subset.
#     """

#     def __init__(
#         self,
#         p_apply: float = 1.0,
#         keep_ratio: Union[float, Tuple[float, float]] = (0.25, 1.0),
#         min_keep: int = 1,
#         seed: Optional[int] = None,
#         metainfo_key: str = "channel_dropout",
#     ):
#         self.p_apply = float(p_apply)
#         self.keep_ratio = keep_ratio
#         self.min_keep = int(min_keep)
#         self.metainfo_key = metainfo_key

#         self._gen = torch.Generator()
#         self._seeded = seed is not None
#         if self._seeded:
#             self._gen.manual_seed(int(seed))

#     def _rand(self, device: torch.device) -> float:
#         return torch.rand((), device=device, generator=self._gen if self._seeded else None).item()

#     def _sample_keep_ratio(self, device: torch.device) -> float:
#         kr = self.keep_ratio
#         if isinstance(kr, (tuple, list)):
#             lo, hi = float(kr[0]), float(kr[1])
#             if not (0.0 < lo <= hi <= 1.0):
#                 raise ValueError(f"keep_ratio range must be in (0,1]; got {(lo, hi)}")
#             u = self._rand(device)
#             return lo + (hi - lo) * u
#         r = float(kr)
#         if not (0.0 < r <= 1.0):
#             raise ValueError(f"keep_ratio must be in (0,1]; got {r}")
#         return r

#     def _augment(self, x: torch.Tensor) -> tuple[torch.Tensor, Dict[str, Any]]:
#         if x.ndim not in (5, 6):
#             raise ValueError(f"Expected BZYXC (5D) or BTZYXC (6D), got shape={tuple(x.shape)}")

#         if self.p_apply < 1.0 and self._rand(x.device) >= self.p_apply:
#             return x, {"applied": False}

#         C = int(x.shape[-1])
#         keep_r = self._sample_keep_ratio(x.device)
#         k = max(self.min_keep, int(round(keep_r * C)))
#         k = min(k, C)

#         keep_idx = torch.randperm(C, device=x.device, generator=self._gen if self._seeded else None)[:k]
#         keep_idx = keep_idx.sort().values

#         x_out = x.index_select(dim=-1, index=keep_idx)

#         info = {
#             "applied": True,
#             "keep_ratio": float(keep_r),
#             "C_in": C,
#             "C_out": int(x_out.shape[-1]),
#             "keep_idx": keep_idx,
#         }
#         return x_out, info

#     def __call__(self, data):
#         if torch.is_tensor(data):
#             x_out, _ = self._augment(data)
#             return x_out

#         if isinstance(data, dict):
#             if "data_tensor" not in data:
#                 raise KeyError("ChannelDropout expects dict with key 'data_tensor'.")
#             x = data["data_tensor"]
#             if not torch.is_tensor(x):
#                 raise TypeError(f"data['data_tensor'] must be torch.Tensor; got {type(x)}")

#             x_out, info = self._augment(x)

#             out = dict(data)
#             out["data_tensor"] = x_out

#             meta = dict(out.get("metainfo", {}))
#             meta[self.metainfo_key] = {
#                 "applied": bool(info.get("applied", False)),
#                 "keep_ratio": float(info.get("keep_ratio", 1.0)),
#                 "C_in": int(info.get("C_in", x.shape[-1])),
#                 "C_out": int(info.get("C_out", x_out.shape[-1])),
#                 "keep_idx": info.get("keep_idx", torch.arange(x_out.shape[-1], device=x_out.device)).detach().cpu(),
#             }
#             out["metainfo"] = meta
#             return out

#         raise TypeError(f"ChannelDropout expects torch.Tensor or dict, got {type(data)}")