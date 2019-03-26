import torch.nn as nn
import torch
import numpy as np

from torch.autograd import Variable
from torch import nn

from collections import namedtuple
from torchvision import models
import matplotlib.pyplot as plt
import util
import torch.nn.functional as F
import projection

class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.eps = 1e-9

    def __call__(self, input, target_is_real):
        if target_is_real:
            return -1.*torch.mean(torch.log(input + self.eps))
        else:
            return -1.*torch.mean(torch.log(1 - input + self.eps))

class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=28, n_layers=3):
        super().__init__()

        sequence = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**(n+1), 8)
            stride = 1 if n == n_layers - 1 else 2
            sequence += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(ndf * nf_mult_prev,
                          ndf * nf_mult,
                          kernel_size=4,
                          stride=stride,
                          padding=0),
                nn.BatchNorm2d(ndf*nf_mult),
                nn.Dropout2d(0.5),
                nn.LeakyReLU(0.2, True)
            ]

        sequence += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(ndf * nf_mult,
                      1,
                      kernel_size=4,
                      stride=1,
                      padding=0),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
