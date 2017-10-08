import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import to_var
from LayerNorm1d import LayerNorm1d

class D_Bot(nn.Module):

    def __init__(self, batch_size, nz):
        super(D_Bot, self).__init__()

        norm = LayerNorm1d

        self.batch_size = batch_size

        self.l1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU())
            
        self.l2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU())

        self.l3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU())

        self.l4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=2),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU())
        
        self.l5 = nn.Sequential(
            nn.Linear(512*4*4, 512),
            nn.LeakyReLU())

        self.l1_z = nn.Sequential(
            nn.Linear(nz, 512),
            nn.LeakyReLU())

        self.l6 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512))

        self.o1 = nn.Conv2d(64, 1, kernel_size=5, padding=2, stride=1)
        self.o2 = nn.Conv2d(128, 1, kernel_size=5, padding=2, stride=1)
        self.o3 = nn.Conv2d(256, 1, kernel_size=5, padding=2, stride=1)
        self.o5 = nn.Sequential(nn.Linear(512,1))

    def forward(self, x,z):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        h4 = self.l4(h3)
        h4 = self.l5(h4.view(self.batch_size, -1))

        h1z = self.l1_z(z*0.0)

        h5 = self.l6(torch.cat((h4, h1z), 1))

        y1 = self.o1(h1)
        y2 = self.o2(h2)
        y3 = self.o3(h3)
        y5 = self.o5(h5)

        return [y1,y2,y3,y5]






