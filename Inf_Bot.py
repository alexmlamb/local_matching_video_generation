import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import to_var
from LayerNorm1d import LayerNorm1d

class Inf_Bot(nn.Module):

    def __init__(self, batch_size, nx, nh, nz):
        super(Inf_Bot, self).__init__()

        self.batch_size = batch_size
        self.nz = nz
        self.l1 = nn.Linear(nx, nh)
        self.a1 = nn.LeakyReLU(0.01)
        self.l2 = nn.Linear(nh,nh)
        self.a2 = nn.LeakyReLU(0.01)
        self.l3_mu = nn.Linear(nh, nz)
        self.l3_sig = nn.Linear(nh, nz)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.l1(x)
        out = self.a1(out)
        out = self.l2(out)
        out = self.a2(out)
        out_mu = self.l3_mu(out)
        out_sig = self.sigmoid(self.l3_sig(out))

        out = out_mu + out_sig * to_var(torch.randn(self.batch_size, self.nz))

        return out



class Inf_Bot_Conv(nn.Module):

    def __init__(self, batch_size, nz):
        super(Inf_Bot_Conv, self).__init__()

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




class Inf_Bot_Conv32(nn.Module):

    def __init__(self, batch_size, nz):
        super(Inf_Bot_Conv32, self).__init__()

        self.batch_size = batch_size

        self.l1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02),
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02),
            nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02))

        self.l2 = nn.Sequential(
            nn.Linear(256*4*4, nz))

    def forward(self, x):
        out = self.l1(x)
        out = out.view(self.batch_size, -1)
        out = self.l2(out)
        return out





