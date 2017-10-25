import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import to_var
from LayerNorm1d import LayerNorm1d

class Gen_Top(nn.Module):

    def __init__(self, batch_size, nz, nh, no):
        super(Gen_Top, self).__init__()

        norm = LayerNorm1d

        self.batch_size = batch_size
        self.nz = nz
        self.no = no
        self.l1 = nn.Linear(nz*2, nh)
        self.bn1 = norm(nh)
        self.a1 = nn.LeakyReLU(0.2)
        self.l2 = nn.Linear(nh,nh)
        self.bn2 = norm(nh)
        self.a2 = nn.LeakyReLU(0.2)
        self.l3 = nn.Linear(nh, no)
        self.l3_sigma = nn.Linear(nh, no)


    def forward(self, z):
        extra_noise = to_var(torch.randn(self.batch_size, self.nz))
        z = torch.cat((extra_noise, z), 1)
        out = self.l1(z)
        out = self.bn1(out)
        out = self.a1(out)
        out = self.l2(out)
        out = self.bn2(out)
        out = self.a2(out)

        out_mu = self.l3(out)

        out_final = out_mu

        return out_final

#Maps from nz to (512,4,4)
class Gen_Top_fc(nn.Module):
    def __init__(self, batch_size, nz, no):
        super(Gen_Top_fc, self).__init__()

        self.batch_size = batch_size
        self.l1 = nn.Sequential(
            nn.Linear(nz, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.02),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.02),
            nn.Linear(1024, no),
            nn.LeakyReLU(0.02))

    def forward(self, z):

        out = self.l1(z)

        return out


#Maps from nz to (512,4,4)
class Gen_Top_4(nn.Module):
    def __init__(self, batch_size, nz):
        super(Gen_Top_4, self).__init__()

        self.batch_size = batch_size
        self.nz = nz
        self.l1 = nn.Sequential(
            nn.Linear(nz, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.02),
            nn.Linear(nz, 128*4*4))

        self.l2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02),
            nn.Conv2d(256, 512*5, kernel_size=1, padding=0, stride=1))

        def forward(self, z):


            out = self.l1(z).view(128,128,4,4)

            out = self.l2(out)

            return out









