#! /usr/bin/python

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import accuracy


''' Adapted from https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/models/projections.py
'''
class LossFunction(nn.Module):
    r"""Implement of sphereface2 for speaker verification:
        Reference:
            [1] Exploring Binary Classification Loss for Speaker Verification
            https://ieeexplore.ieee.org/abstract/document/10094954
            [2] Sphereface2: Binary classification is all you need for deep face recognition
            https://arxiv.org/abs/2108.01513
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            scale: norm of input feature
            margin: margin
            lanbuda: weight of positive and negative pairs
            t: parameter for adjust score distribution
            margin_type: A:cos(theta+margin) or C:cos(theta)-margin
        Recommend margin:
            training: 0.2 for C and 0.15 for A
            LMF: 0.3 for C and 0.25 for A
        """

    def __init__(self,
                 embedding_dim,
                 num_classes,
                 scale=32.0,
                 margin=0.2,
                 lanbuda=0.7,
                 t=3,
                 margin_type='C',
                 **kwargs):
        super(LossFunction, self).__init__()
        self.in_feats = embedding_dim
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(1, 1))
        self.t = t
        self.lanbuda = lanbuda
        self.margin_type = margin_type

        ########
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)
        ########

        print('Initialised %s-SphereFace2 margin %.2f scale %.2f lanbuda %.2f t %.2f' % (margin_type, self.margin, self.scale, self.lanbuda, self.t))

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)

    def fun_g(self, z, t: int):
        gz = 2 * torch.pow((z + 1) / 2, t) - 1
        return gz

    def forward(self, x, label):
        assert len(x.shape) == 3
        label = label.repeat_interleave(x.shape[1])
        x = x.reshape(-1, self.in_feats)
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        # compute similarity
        cos = F.linear(F.normalize(x), F.normalize(self.weight))

        if self.margin_type == 'A':  # arcface type
            sin = torch.sqrt(1.0 - torch.pow(cos, 2))
            cos_m_theta_p = self.scale * self.fun_g(torch.where(cos > self.th, 
                            cos * self.cos_m - sin * self.sin_m, cos - self.mmm), self.t) + self.bias[0][0]
            cos_m_theta_n = self.scale * self.fun_g(cos * self.cos_m + sin * self.sin_m, self.t) + self.bias[0][0]
            cos_p_theta = self.lanbuda * torch.log(1 + torch.exp(-1.0 * cos_m_theta_p))
            cos_n_theta = (1 - self.lanbuda) * torch.log(1 + torch.exp(cos_m_theta_n))
        else:  # cosface type
            cos_m_theta_p = self.scale * (self.fun_g(cos, self.t) - self.margin) + self.bias[0][0]
            cos_m_theta_n = self.scale * (self.fun_g(cos, self.t) + self.margin) + self.bias[0][0]
            cos_p_theta = self.lanbuda * torch.log(1 + torch.exp(-1.0 * cos_m_theta_p))
            cos_n_theta = (1 - self.lanbuda) * torch.log(1 + torch.exp(cos_m_theta_n))

        target_mask = x.new_zeros(cos.size())
        target_mask.scatter_(1, label.view(-1, 1).long(), 1.0)
        nontarget_mask = 1 - target_mask
        cos1 = (cos - self.margin) * target_mask + cos * nontarget_mask
        output = self.scale * cos1  # for computing the accuracy
        loss = (target_mask * cos_p_theta + nontarget_mask * cos_n_theta).sum(1).mean()
        
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1
