import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import to_var

class D_Top(nn.Module):

    def __init__(self, batch_size, nb, nt, nh):
        super(D_Top, self).__init__()

        self.batch_size = batch_size

        self.lbot = nn.Linear(nb, nh)
        self.abot = nn.LeakyReLU(0.2)

        self.ltop = nn.Linear(nt, nh)
        self.atop = nn.LeakyReLU(0.2)

        self.l1 = nn.Linear(nh+nh, nh)
        self.a1 = nn.LeakyReLU(0.2)
        self.l2 = nn.Linear(nh,nh)
        self.a2 = nn.LeakyReLU(0.2)
        self.l3 = nn.Linear(nh, 1)

    def forward(self, z_bot, z_top=None):

        h_bot = self.lbot(z_bot)
        h_bot = self.abot(h_bot)

        h_top = self.ltop(z_top)
        h_top = self.atop(h_top)

        h = torch.cat((h_bot,h_top), 1)

        out = self.l1(h)
        out = self.a1(out)
        out = self.l2(out)
        out = self.a2(out)
        out = self.l3(out)
        return [out]






