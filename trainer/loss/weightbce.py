import torch
import torch.nn as nn


class LossFunction(nn.Module):
    def __init__(self, pos_weight=0.5):
        super(LossFunction, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, y_pred, y_true):
        # Clip predictions to prevent log(0)
        y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)

        # Calculate weighted BCE loss
        loss_1 = -self.pos_weight * y_true * torch.log(y_pred)
        loss_0 = -(1-self.pos_weight) * (1 - y_true) * torch.log(1 - y_pred)
        loss = loss_1 + loss_0

        return torch.mean(loss)

