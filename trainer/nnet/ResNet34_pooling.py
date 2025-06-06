#! /usr/bin/python
# -*- encoding: utf-8 -*-
'''
Fast ResNet
https://arxiv.org/pdf/2003.11982.pdf
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
try:
    from .pooling import *
except:
    from pooling import *

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_filters, embedding_dim, n_mels=80, pooling_type="TSP", **kwargs):
        super(ResNet, self).__init__()

        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=3, stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(2, 2))

        out_dim = num_filters[3] * block.expansion * (n_mels // 8)

        if pooling_type == "Temporal_Average_Pooling" or pooling_type == "TAP":
            self.pooling = Temporal_Average_Pooling()
            self.bn2 = nn.BatchNorm1d(out_dim)
            self.fc = nn.Linear(out_dim, embedding_dim)
            self.bn3 = nn.BatchNorm1d(embedding_dim)

        elif pooling_type == "Temporal_Statistics_Pooling" or pooling_type == "TSP":
            self.pooling = Temporal_Statistics_Pooling()
            self.bn2 = nn.BatchNorm1d(out_dim * 2)
            self.fc = nn.Linear(out_dim * 2, embedding_dim)
            self.bn3 = nn.BatchNorm1d(embedding_dim)

        elif pooling_type == "Self_Attentive_Pooling" or pooling_type == "SAP":
            self.pooling = Self_Attentive_Pooling(out_dim)
            self.bn2 = nn.BatchNorm1d(out_dim)
            self.fc = nn.Linear(out_dim, embedding_dim)
            self.bn3 = nn.BatchNorm1d(embedding_dim)

        elif pooling_type == "Attentive_Statistics_Pooling" or pooling_type == "ASP":
            self.pooling = Attentive_Statistics_Pooling(out_dim)
            self.bn2 = nn.BatchNorm1d(out_dim * 2)
            self.fc = nn.Linear(out_dim * 2, embedding_dim)
            self.bn3 = nn.BatchNorm1d(embedding_dim)

        else:
            raise ValueError('{} pooling type is not defined'.format(pooling_type))


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(x.shape[0], -1, x.shape[-1])

        x = self.pooling(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.bn3(x)
        return x


def Speaker_Encoder(embedding_dim=256, **kwargs):
    # Number of filters
    num_filters = [32, 64, 128, 256]
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_filters, embedding_dim, **kwargs)
    return model

if __name__ == '__main__':
    model = Speaker_Encoder()
    total = sum([param.nelement() for param in model.parameters()])
    print(total/1e6)
    data = torch.randn(10, 80, 100)
    out = model(data)
    print(data.shape)
    print(out.shape)

