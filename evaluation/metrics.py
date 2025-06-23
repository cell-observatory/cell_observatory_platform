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