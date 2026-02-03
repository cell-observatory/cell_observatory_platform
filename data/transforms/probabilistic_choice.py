import random
from typing import Any, Dict, List

from hydra.utils import instantiate
from omegaconf import DictConfig


class ProbabilisticChoice:
    """
    Meta-transform that randomly selects one transform from a list based on probabilities.

    This allows creating probabilistic pipelines where different transforms are applied
    with different frequencies.

    Example:
        ProbabilisticChoice(
            transforms=[resize_transform, crop_transform],
            probs=[0.5, 0.5],
        )
    """

    def __init__(
        self,
        transforms: List[Any],
        probs: List[float],
    ) -> None:
        """
        Args:
            transforms: List of transform configs (DictConfig) or instantiated transforms.
            probs: List of probabilities for each transform. Must sum to 1.0.
        """
        if len(transforms) != len(probs):
            raise ValueError(
                f"transforms and probs must have same length, "
                f"got {len(transforms)} and {len(probs)}"
            )
        total = sum(probs)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"probs must sum to 1.0, got {total}")

        self.transforms = []
        for t in transforms:
            if isinstance(t, DictConfig):
                self.transforms.append(instantiate(t))
            else:
                self.transforms.append(t)

        self.probs = probs

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a randomly selected transform to the data.

        Args:
            data: Input data dict

        Returns:
            Transformed data dict
        """
        # Sample transform index based on probabilities
        r = random.random()
        cumsum = 0.0
        idx = len(self.transforms) - 1

        for i, p in enumerate(self.probs):
            cumsum += p
            if r < cumsum:
                idx = i
                break

        return self.transforms[idx](data)