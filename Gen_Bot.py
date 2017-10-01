import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import to_var

class Gen_Bot(nn.Module):

    def __init__(self, batch_size, nz, nh, no):
        super(Gen_Bot, self).__init__()

        self.batch_size = batch_size
        self.nz = nz
        self.l1 = nn.Linear(nz*2, nh)
        self.bn1 = nn.BatchNorm1d(nh)
        self.a1 = nn.LeakyReLU(0.2)
        self.l2 = nn.Linear(nh,nh)
        self.bn2 = nn.BatchNorm1d(nh)
        self.a2 = nn.LeakyReLU(0.2)
        self.l3 = nn.Linear(nh, no)
        self.a3 = nn.Tanh()

    def forward(self, z):
        extra_noise = to_var(torch.randn(self.batch_size, self.nz))
        z = torch.cat((extra_noise, z), 1)
        out = self.l1(z)
        out = self.bn1(out)
        out = self.a1(out)
        out = self.l2(out)
        out = self.bn2(out)
        out = self.a2(out)
        out = self.l3(out)
        out = self.a3(out)
        return out







