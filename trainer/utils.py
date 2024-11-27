#!/usr/bin/env python
# encoding: utf-8

import random
import numpy as np
import torch
from torch.nn import functional as F

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        assert len(inputs.size()) == 2, 'The number of dimensions of inputs tensor must be 2!'
        # reflect padding to match lengths of in/out
        inputs = inputs.unsqueeze(1)
        inputs = F.pad(inputs, (1, 0), 'reflect')
        return F.conv1d(inputs, self.flipped_filter).squeeze(1)


def load_checkpoint(model: torch.nn.Module, path: str, keep_loss_weight=True):
    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint["state_dict"]
    if not keep_loss_weight:
        state_dict.pop('loss.weight')
    model.load_state_dict(state_dict, strict=False)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

