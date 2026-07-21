# import math
# import random
# from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# import torch
# from hydra.utils import instantiate
# from omegaconf import DictConfig

# from cell_observatory_platform.data.transforms.utils import stack_metainfo


# class MultiCrop3D:
#     """
#     Meta-augmentation that generates multiple named crop streams from a single input.

#     Each stream has a crop transform and a count specifying how many independent
#     crops to produce. All crops within a stream are concatenated along the batch
#     dimension so downstream code sees (B * count, Z', Y', X', C).

#     Input:
#       dict{data_tensor: Tensor(B, Z, Y, X, C), metainfo: dict}

#     Output:
#       dict{
#         data_tensor: {name: Tensor(B*count, Z', Y', X', C), ...},
#         metainfo:    {name: {n_crops: int, ...}, ...}
#       }
#     """

#     def __init__(
#         self,
#         crop_transforms: List[Any],  # Crop instances or DictConfig
#         names: Optional[List[str]] = None,  # stream names aligned with crop_transforms
#         counts: Optional[List[int]] = None,  # number of crops per stream
#     ) -> None:
#         """
#         Args:
#             crop_transforms: List of crop transform configs (DictConfig) or instantiated crop transforms.
#             names: List of stream names aligned with crop_transforms.
#             counts: Number of independent crops to generate per stream.
#                     Each crop is produced by running the transform independently
#                     (with fresh random state) on the same input.
#                     Defaults to [1, 1, ...] if not provided.
#         """
#         if not crop_transforms:
#             raise ValueError("MultiCrop3D requires at least one crop transform.")
#         self.crop_transforms: List[Callable] = []
#         for c in crop_transforms:
#             self.crop_transforms.append(instantiate(c) if isinstance(c, DictConfig) else c)

#         if names is None:
#             names = [f"crop{i}" for i in range(len(self.crop_transforms))]
#         if len(names) != len(self.crop_transforms):
#             raise ValueError(f"names must match crop_transforms length; got {len(names)} vs {len(self.crop_transforms)}")

#         if len(set(names)) != len(names):
#             raise ValueError(f"names must be unique; got {names}")

#         self.names = names

#         if counts is None:
#             counts = [1] * len(self.crop_transforms)
#         if len(counts) != len(self.crop_transforms):
#             raise ValueError(f"counts must match crop_transforms length; got {len(counts)} vs {len(self.crop_transforms)}")
#         if any(c < 1 for c in counts):
#             raise ValueError(f"All counts must be >= 1; got {counts}")

#         self.counts = counts

#     def __call__(self, data: Union[torch.Tensor, Dict[str, Any]]) -> Union[torch.Tensor, Dict[str, Any]]:
#         if isinstance(data, dict):
#             if "data_tensor" not in data:
#                 raise KeyError("MultiCrop3D expects 'data_tensor' in input dict")
#             data_dict = {"data_tensor": data["data_tensor"], "metainfo": data.get("metainfo", {}) or {}}
#         else:
#             raise TypeError(f"MultiCrop3D expects dict; got {type(data)}")

#         x = data_dict["data_tensor"]
#         meta = data_dict["metainfo"]
#         if not torch.is_tensor(x):
#             raise TypeError(f"MultiCrop3D expected data_tensor Tensor; got {type(x)}")

#         out_tensors: Dict[str, torch.Tensor] = {}
#         out_meta: Dict[str, Dict[str, Any]] = {}

#         for name, crop_transform, count in zip(self.names, self.crop_transforms, self.counts):
#             crops: List[torch.Tensor] = []
#             meta_list: List[Dict[str, Any]] = []

#             for _ in range(count):
#                 # Each crop gets an independent copy of meta so transforms
#                 # don't interfere with each other across crops
#                 data_dict_i = {"data_tensor": x, "metainfo": dict(meta)}

#                 out_i = crop_transform(data_dict_i)

#                 if not isinstance(out_i, dict) or "data_tensor" not in out_i:
#                     raise TypeError(
#                         f"Crop transform for stream {name!r} must return dict with 'data_tensor'; "
#                         f"got {type(out_i)}"
#                     )
#                 crops.append(out_i["data_tensor"])
#                 meta_list.append(out_i.get("metainfo", {}))

#             # Concatenate all crops along the batch dimension:
#             # each crop is (B, Z', Y', X', C) -> result is (B * count, Z', Y', X', C)
#             out_tensors[name] = torch.cat(crops, dim=0)

#             # Merge per-crop metainfo to maintain any 1-1 mapping between
#             # samples in data_tensor and entries in metainfo (lists concatenated,
#             # tensors cat'd along dim 0, dicts recursed, scalars kept as-is).
#             stream_meta = stack_metainfo(meta_list)
#             stream_meta["n_crops"] = count
#             out_meta[name] = stream_meta

#         return {"data_tensor": out_tensors, "metainfo": out_meta}