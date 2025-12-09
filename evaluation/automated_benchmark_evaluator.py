from typing import Dict, List, Literal, Optional

import torch

from cell_observatory_platform.evaluation.evaluator import DatasetEvaluator
from cell_observatory_platform.evaluation.metrics import MAEMetric, NRMSEMetric  # SSIMMetric,


class AutomatedBenchmarkEvaluator(DatasetEvaluator):
    def __init__(
        self,
        metric_reductions: List[Dict[str, str]],
        pred_key: Optional[str] = None,
        target_key: str = "data_tensor",
        ssim_data_range: Optional[float] = 1.0,
        ssim_kernel_size: int = 11,
        ssim_sigma: float = 1.5,
        ssim_K1: float = 0.01,
        ssim_K2: float = 0.03,
        reduction: Literal["elementwise_mean", "sum"] = "elementwise_mean",
    ):
        self.metrics = {}
        for spec in metric_reductions:
            for name, reduce_op in spec.items():
                lname = name.lower()
                if lname == "ssim":
                    pass
                    # self.metrics[name] = SSIMMetric(
                    #     data_range=ssim_data_range,
                    #     kernel_size=ssim_kernel_size,
                    #     sigma=ssim_sigma,
                    #     K1=ssim_K1,
                    #     K2=ssim_K2,
                    #     reduce_method=reduce_op,
                    #     reduction=reduction,
                    # )
                elif lname in ("nrmse", "norm_rmse", "normalized_rmse"):
                    self.metrics[name] = NRMSEMetric(reduce_method=reduce_op)
                elif lname in ("mae", "l1"):
                    self.metrics[name] = MAEMetric(reduce_method=reduce_op)
                else:
                    raise ValueError(f"Unknown metric name: {name}")

        self.pred_key = pred_key
        self.target_key = target_key

        self._results = {name: None for name in self.metrics.keys()}

    def reset(self):
        for m in self.metrics.values():
            m.reset()
        self._results = {k: None for k in self._results.keys()}

    @torch.no_grad()
    def process(self, data_sample, outputs, loss_dict):
        if self.pred_key is None:
            pred = outputs
        else:
            pred = outputs[self.pred_key]

        target = data_sample["metainfo"][self.target_key][0]

        if pred.dtype != torch.float32:
            pred = pred.float()
        if target.dtype != torch.float32:
            target = target.float()

        target = target.to(device=pred.device, dtype=pred.dtype)

        for name, metric_impl in self.metrics.items():
            metric_impl(pred, target, None)

    def evaluate(self):
        for name, metric_impl in self.metrics.items():
            self._results[name] = float(metric_impl.aggregate())
        return self._results
