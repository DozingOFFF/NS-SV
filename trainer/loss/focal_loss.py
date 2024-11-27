import torch
import torch.nn as nn

class Focal_Loss(nn.Module):
    """
    二分类Focal Loss
    """
    def __init__(self, alpha=0.5, gamma=1):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        """
        input:sigmoid的输出结果
        target：标签
        """
        eps = 1e-7
        loss_1 = -1 * self.alpha * torch.pow((1 - input), self.gamma) * torch.log(input + eps) * target
        loss_0 = -1 * (1 - self.alpha) * torch.pow(input, self.gamma) * torch.log(1 - input + eps) * (1 - target)
        loss = loss_0 + loss_1
        return torch.mean(loss)