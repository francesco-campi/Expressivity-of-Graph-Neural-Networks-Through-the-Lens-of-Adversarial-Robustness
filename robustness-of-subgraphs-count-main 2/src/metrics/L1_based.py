from torch.nn import L1Loss
from torch import Tensor, ones, abs, mean

class L1LossStd(L1Loss):
    """Metric that computes the L1 loss divided by the standard deviation of the ground truths."""
    def __init__(self, std: Tensor) -> None:
        self.std = std
        super().__init__()

    def forward(self, prediction: Tensor, ground_truth: Tensor):
        return super().forward(prediction, ground_truth) / self.std

class L1LossCount(L1Loss):
    """Metric that computes the L1 loss divided by the ground truth."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, prediction: Tensor, ground_truth: Tensor):
        l1_loss = abs(prediction - ground_truth)
        l1_loss = l1_loss / (ground_truth + 1) 
        return mean(l1_loss)
