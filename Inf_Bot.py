import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import to_var
from LayerNorm1d import LayerNorm1d

class Inf_Bot(nn.Module):

    def __init__(self, batch_size, nz):
        super(Inf_Bot, self).__init__()

        norm = LayerNorm1d

        self.batch_size = batch_size

        self.l1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2, stride=2),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2))

        self.l2 = nn.Sequential(
            nn.Linear(512*4*4, nz))

    def forward(self, z):
        out = self.l1(z)
        out = out.view(self.batch_size, -1)
        out = self.l2(out)
        return out






