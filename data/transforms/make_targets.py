import torch

# Semantic-stage targets are Form D (role-keyed dict of batched maps) -- see
# data/data_types.py for the Form D / Form S nomenclature.


def _source_slice(targets: dict, role: str) -> torch.Tensor:
    """The batched ``(B, D, H, W)`` map published under ``role``. Fails hard if the
    role isn't in the Form-D dict."""
    if role not in targets:
        raise KeyError(f"targets {list(targets)} has no source role {role!r}")
    return targets[role]


def _append_map(targets: dict, m: torch.Tensor, role: str) -> None:
    """Publish a derived batched ``(B, D, H, W)`` map under a new role key."""
    targets[role] = m


class DeepCopyInputsAsTargets:
    # Snapshots the image as a REGRESSION TARGET (the clean reference for
    # denoising), so the counts it copies are a label rather than an
    # intermediate: quantizing them degrades what the loss is measured against.
    reads_raw_counts = True

    def __init__(self, role: str = "denoising"):
        # Form-D role the clone is published under; must match the preprocessor's
        # recon_role (both default to "denoising" so stock configs can't drift).
        self.role = role

    def __call__(self, data: dict) -> dict:
        if not isinstance(data, dict):
            raise ValueError(f"DeepCopyInputsAsTargets expects dict, got {type(data)}")
        if "data_tensor" not in data:
            raise KeyError("DeepCopyInputsAsTargets expects 'data_tensor' in dict.")
        if "metainfo" not in data:
            data["metainfo"] = {}
        # if "targets" in data["metainfo"] and data["metainfo"]["targets"] is not None:
        #     raise ValueError("targets already exists in metainfo")
        data["metainfo"]["targets"] = {self.role: data["data_tensor"].clone()}
        return data

class InstanceToBoundaryMask:
    def __init__(self, source_role: str, connectivity: int = 1):
        # Which Form-D role to derive boundaries from.
        self.source_role = source_role
        self.connectivity = connectivity

        # Build shifts for 3D connectivity:
        # 6-neighborhood: Manhattan distance 1
        # 18-neighborhood: Manhattan distance <= 2 excluding corners (i.e., max(|dz|,|dy|,|dx|)=1 but not all three nonzero)
        # 26-neighborhood: all neighbors in {-1,0,1}^3 except (0,0,0)
        shifts = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == dy == dx == 0:
                        continue
                    manhattan = abs(dz) + abs(dy) + abs(dx)
                    chebyshev = max(abs(dz), abs(dy), abs(dx))

                    if self.connectivity == 1:
                        if manhattan == 1:
                            shifts.append((dz, dy, dx))
                    elif self.connectivity == 2:
                        # 18-neighborhood: chebyshev==1 but exclude 3-axis corners (manhattan==3)
                        if chebyshev == 1 and manhattan <= 2:
                            shifts.append((dz, dy, dx))
                    elif self.connectivity == 3:
                        if chebyshev == 1:
                            shifts.append((dz, dy, dx))
                    else:
                        raise ValueError("connectivity must be 1, 2, or 3 for 3D")

        self.shifts = shifts

    def _one_hot_encode_labels(self, labels: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(labels, num_classes=int(labels.max() + 1))

    def __call__(self, data: dict) -> dict:
        # Read the batched source role, compute boundaries batched, publish the
        # derived map under its own role -- one assignment, no per-sample loop.
        targets = data["metainfo"]["targets"]
        src = _source_slice(targets, self.source_role).to(torch.int32)
        boundary = self._instance_to_boundary_mask(src)  # (B, D, H, W) bool
        _append_map(targets, boundary, "boundary")
        return data


    def _instance_to_boundary_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """
        labels: (B,D,H,W) int (instance/semantic labels; background can be 0).
        Returns: bool mask of same shape, True where boundaries occur.
        connectivity=1 -> 6-neighborhood, 2 -> 18, 3 -> 26 (3D). [web:1]
        """
        if labels.dim() != 4:
            raise ValueError(f"labels must be a 4D tensor assumed to be (B,D,H,W), got {labels.shape}")

        B, D, H, W = labels.shape
        x = labels

        boundary = torch.zeros((B, D, H, W), dtype=torch.bool, device=x.device)

        for dz, dy, dx in self.shifts:
            z0, z1 = max(0, dz), D + min(0, dz)
            y0, y1 = max(0, dy), H + min(0, dy)
            x0, x1 = max(0, dx), W + min(0, dx)

            a = x[:, z0:z1, y0:y1, x0:x1]
            b = x[:, z0-dz:z1-dz, y0-dy:y1-dy, x0-dx:x1-dx]
            boundary[:, z0:z1, y0:y1, x0:x1] |= (a != b)

        return boundary


class ForegroundMasks:
    def __init__(self, source_role: str, remove_boundary: bool = True):
        # Which Form-D role to derive foreground from.
        self.source_role = source_role
        self.remove_boundary = remove_boundary

    def __call__(self, data: dict) -> dict:
        # Read the batched source role (and the derived "boundary" role when
        # removing boundaries), compute foreground batched, publish under its role.
        targets = data["metainfo"]["targets"]
        src = _source_slice(targets, self.source_role).to(torch.int32)
        if self.remove_boundary:
            # TODO: rename once DB roles table lands.
            bm = _source_slice(targets, "boundary")  # (B, D, H, W)
            foreground = self._foreground_masks(src, bm)
        else:
            foreground = self._foreground_masks(src)
        # TODO: rename once DB roles table lands.
        _append_map(targets, foreground, "foreground")
        return data

    def _foreground_masks(self, labels: torch.Tensor, boundary_masks: torch.Tensor | None = None) -> torch.Tensor:
        """
        labels: (B,D,H,W) int (instance/semantic labels; background can be 0).
        boundary_masks: (B,D,H,W) bool (boundary masks).
        Returns: bool mask of same shape, True where boundaries occur.
        """
        if labels.dim() != 4:
            raise ValueError(f"labels must be a 4D tensor assumed to be (B,D,H,W), got {labels.shape}")

        B, D, H, W = labels.shape
        x = labels

        foreground_masks = torch.zeros((B, D, H, W), dtype=torch.bool, device=x.device)

        if boundary_masks is not None:
            # boundary_masks is a bool role published by InstanceToBoundaryMask;
            # keep the explicit .bool() so an int-typed map (e.g. loaded from
            # storage) still gets a LOGICAL not, never a bitwise one.
            foreground_masks = (x != 0) & ~boundary_masks.bool()
        else:
            foreground_masks = (x != 0)

        return foreground_masks
