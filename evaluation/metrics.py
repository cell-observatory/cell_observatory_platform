import abc


class Metric(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, outputs, targets, loss):
        pass

    @abc.abstractmethod
    def aggregate(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class TrainLosses(Metric):
    def __init__(self, reduce_method: str = "mean"):
        self.reduce_method = reduce_method
        self.loss_values = []

    def __call__(self, outputs, targets, loss):
        self.loss_values.append(loss.item())

    def aggregate(self):
        assert self.loss_values, "No loss values to aggregate."
        if self.reduce_method == "mean":
            return sum(self.loss_values) / len(self.loss_values)
        elif self.reduce_method == "min":
            return min(self.loss_values)
        elif self.reduce_method == "max":
            return max(self.loss_values)
        else:
            raise ValueError(f"Unknown reduce method: {self.reduce_method}")

    def reset(self):
        self.loss_values.clear()


class ReduceBuffer:
    def __init__(self, reduce_method: str = "mean"):
        self.reduce_method = reduce_method
        self.values: List[float] = []

    def add(self, v: torch.Tensor | float):
        v = float(v.item() if torch.is_tensor(v) else v)
        self.values.append(v)

    def aggregate(self) -> float:
        assert self.values, "No values to aggregate."
        if self.reduce_method == "mean":
            return sum(self.values) / len(self.values)
        elif self.reduce_method == "min":
            return min(self.values)
        elif self.reduce_method == "max":
            return max(self.values)
        else:
            raise ValueError(f"Unknown reduce method: {self.reduce_method}")

    def reset(self):
        self.values.clear()


# class SSIMMetric(Metric):
#     def __init__(self,
#                  data_range: Optional[float] = 1.0,
#                  kernel_size: int = 11,
#                  sigma: float = 1.5,
#                  K1: float = 0.01,
#                  K2: float = 0.03,
#                  reduction: Literal["elementwise_mean", "sum"] = "elementwise_mean",
#                  reduce_method: str = "mean",
#     ):
#         super().__init__()
#         self.data_range = data_range
#         self.kernel_size = kernel_size

#         self.sigma = sigma

#         self.K1 = K1
#         self.K2 = K2

#         self.buf = ReduceBuffer(reduce_method)
#         self.reduction = reduction

#     @torch.no_grad()
#     def __call__(self, outputs, targets, loss=None):
#         assert outputs.shape == targets.shape, f"SSIM: mismatched shapes {outputs.shape} vs {targets.shape}"
#         outputs, targets = _ssim_check_inputs(outputs, targets)
#         ssim_val = _ssim_update(
#             outputs,
#             targets,
#             data_range=self.data_range,
#             kernel_size=self.kernel_size,
#             sigma=self.sigma,
#             K1=self.K1,
#             K2=self.K2,
#             nonnegative_ssim=True,
#             full=False,
#         )

#         if self.reduction == "elementwise_mean":
#             ssim_val = ssim_val / outputs.shape[0]

#         self.buf.add(ssim_val)

#     def aggregate(self):
#         return self.buf.aggregate()

#     def reset(self):
#         self.buf.reset()


class NRMSEMetric(Metric):
    def __init__(self, reduce_method: str = "mean", eps: float = 1e-8):
        self.buf = ReduceBuffer(reduce_method)
        self.eps = eps

    @torch.no_grad()
    def __call__(self, outputs, targets, loss=None):
        x = outputs.to(dtype=torch.float32)
        y = targets.to(dtype=torch.float32)
        diff = x - y
        mse = (diff * diff).mean()
        rmse = torch.sqrt(mse)
        denom = (torch.amax(y) - torch.amin(y)).clamp_min(self.eps)
        nrmse = rmse / denom
        self.buf.add(nrmse)

    def aggregate(self):
        return self.buf.aggregate()

    def reset(self):
        self.buf.reset()


class MAEMetric(Metric):
    def __init__(self, reduce_method: str = "mean"):
        self.buf = ReduceBuffer(reduce_method)

    @torch.no_grad()
    def __call__(self, outputs, targets, loss=None):
        x = outputs.to(dtype=torch.float32)
        y = targets.to(dtype=torch.float32)
        mae = (x - y).abs().mean()
        self.buf.add(mae)

    def aggregate(self):
        return self.buf.aggregate()

    def reset(self):
        self.buf.reset()
