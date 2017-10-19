import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import to_var

class D_Bot(nn.Module):

    def __init__(self, batch_size, nx, nz, nh):
        super(D_Bot, self).__init__()

        self.batch_size = batch_size

        self.lbot = nn.Linear(nx, nh)
        self.abot = nn.LeakyReLU(0.02)

        self.ltop = nn.Linear(nz, nh)
        self.atop = nn.LeakyReLU(0.02)

        self.l1 = nn.Linear(nh+nh, nh)
        self.a1 = nn.LeakyReLU(0.02)
        self.l2 = nn.Linear(nh,nh)
        self.a2 = nn.LeakyReLU(0.02)
        self.l3 = nn.Linear(nh, 1)

    def forward(self, x, z):

        h_bot = self.lbot(x)
        h_bot = self.abot(h_bot)

        h_top = self.ltop(z)
        h_top = self.atop(h_top)

        h = torch.cat((h_bot,h_top), 1)

        out = self.l1(h)
        out = self.a1(out)
        out = self.l2(out)
        out = self.a2(out)
        out = self.l3(out)

        return [out,h_bot]


class D_Bot_Conv32(nn.Module):
    def __init__(self, batch_size, nz):
        super(D_Bot_Conv32, self).__init__()
        self.batch_size = batch_size

        self.zo2 = nn.Sequential(
            nn.Linear(nz, 512),
            nn.LeakyReLU(0.02),
            nn.Linear(512, 256*8*8),
            nn.LeakyReLU(0.02))

        self.zo3 = nn.Sequential(
            nn.Linear(nz, 512),
            nn.LeakyReLU(0.02),
            nn.Linear(512, 512*4*4),
            nn.LeakyReLU(0.02))

        self.l1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.02))
        self.l2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.02))

        self.l3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.02))

        self.l_end = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=5, padding=2, stride=2))

    def forward(self, x,z):

        zo2 = self.zo2(z).view(self.batch_size,256,8,8) #goes to 256x8x8
        zo3 = self.zo3(z).view(self.batch_size,512,4,4)

        out = self.l1(x)
        out = self.l2(out) + zo2
        out = self.l3(out) + zo3
        out = self.l_end(out)
        out = out.view(self.batch_size,-1)
        return out










