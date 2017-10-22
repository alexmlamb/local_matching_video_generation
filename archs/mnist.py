import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from operator import mul
# import numpy as np
# from utils import to_var
# from LayerNorm1d import LayerNorm1d

NUM_KERNELS = [64, 128]
KERNEL_SIZE = [3, 3]
PADDING = [2, 1]
STRIDE = [2, 2]
NUM_CHANNELS = 1

def compute_conv_output_size(input_size, kernel_size, padding, stride):
    return (input_size - kernel_size + 2 * padding) / stride + 1


class Inf_Low(nn.Module):
    def __init__(self, batch_size, nx, nz):
        super(Inf_Low, self).__init__()
        self.batch_size = batch_size

        nconv1 = compute_conv_output_size(nx, KERNEL_SIZE[0], PADDING[0], STRIDE[0])
        nconv2 = compute_conv_output_size(nconv1, KERNEL_SIZE[1], PADDING[1], STRIDE[1])
        conv_out_shape = [NUM_KERNELS[-1], nconv2, nconv2]
        fc_input_size = reduce(mul, conv_out_shape, 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, NUM_KERNELS[0], kernel_size=KERNEL_SIZE[0],
                      padding=PADDING[0], stride=STRIDE[0]),
            nn.LeakyReLU(0.2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(NUM_KERNELS[0], NUM_KERNELS[1], kernel_size=KERNEL_SIZE[1],
                      padding=PADDING[1], stride=STRIDE[1]),
            nn.LeakyReLU(0.2))

        self.fc = nn.Linear(fc_input_size, nz)

    def forward(self, x):
        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h_flattened = h2.view(self.batch_size, -1)
        out = self.fc(h_flattened)
        return out

    def get_conv_out_shape():
        return self.conv_out_shape


class Gen_Low(nn.Module):

    def __init__(self, batch_size, nx, nz):
        super(Gen_Low, self).__init__()
        self.batch_size = batch_size

        nconv1 = compute_conv_output_size(nx, KERNEL_SIZE[0], PADDING[0], STRIDE[0])
        nconv2 = compute_conv_output_size(nconv1, KERNEL_SIZE[1], PADDING[1], STRIDE[1])
        self.conv_kernel_len = nconv2
        conv_out_shape = [NUM_KERNELS[-1], nconv2, nconv2]
        fc_output_size = reduce(mul, conv_out_shape, 1)

        self.fc = nn.Linear(nz, fc_output_size)

        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(NUM_KERNELS[-1], NUM_KERNELS[-2], kernel_size=KERNEL_SIZE[-1],
                               padding=PADDING[-1], stride=STRIDE[-1]),
            nn.LeakyReLU(0.2))

        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(NUM_KERNELS[-2], NUM_CHANNELS, kernel_size=KERNEL_SIZE[-2],
                               padding=PADDING[-2], stride=STRIDE[-2]),
            nn.Tanh())

    def forward(self, z):
        h1 = self.fc(z)
        h1_2d = h1.view(self.batch_size, NUM_KERNELS[-1], self.conv_kernel_len, self.conv_kernel_len)
        h_conv1 = self.convT1(h1_2d)
        out = self.convT2(h_conv1)
        return out


class Disc_Low(nn.Module):
    def __init__(self, batch_size, nz):
        super(Disc_Low, self).__init__()
        self.batch_size = batch_size

        # self.zo2 = nn.Sequential(
        #     nn.Linear(nz, 512),
        #     nn.LeakyReLU(0.02),
        #     nn.Linear(512, 256*8*8),
        #     nn.LeakyReLU(0.02))
        # 
        # self.zo3 = nn.Sequential(
        #     nn.Linear(nz, 512),
        #     nn.LeakyReLU(0.02),
        #     nn.Linear(512, 512*4*4),
        #     nn.LeakyReLU(0.02))

        self.l1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.02))
        self.l2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.02))

        self.l3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.02))

        self.l_end = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=5, padding=2, stride=2))

    def forward(self, x):
    # def forward(self, x, z):

        # zo2 = self.zo2(z).view(self.batch_size,256,8,8) #goes to 256x8x8
        # zo3 = self.zo3(z).view(self.batch_size,512,4,4)

        out = self.l1(x)
        # out = self.l2(out) + zo2
        # out = self.l3(out) + zo3
        out = self.l2(out)
        out = self.l3(out)
        out = self.l_end(out)
        out = out.view(self.batch_size,-1)
        return out


class Disc_High(nn.Module):

    def __init__(self, batch_size, nb, nt, nh):
        super(Disc_High, self).__init__()

        self.batch_size = batch_size

        # self.lbot = nn.Linear(nb, nh)
        # self.abot = nn.LeakyReLU(0.2)
        # 
        # self.ltop = nn.Linear(nt, nh)
        # self.atop = nn.LeakyReLU(0.2)

        self.l1 = nn.Linear(nb, nh)
        # self.l1 = nn.Linear(nh+nh, nh)
        self.a1 = nn.LeakyReLU(0.2)
        self.l2 = nn.Linear(nh,nh)
        self.a2 = nn.LeakyReLU(0.2)
        self.l3 = nn.Linear(nh, 1)

    def forward(self, z_bot, z_top=None):

        # h_bot = self.lbot(z_bot)
        # h_bot = self.abot(h_bot)
        # 
        # h_top = self.ltop(z_top)
        # h_top = self.atop(h_top)
        # 
        # h = torch.cat((h_bot,h_top), 1)

        h = z_bot

        out = self.l1(h)
        out = self.a1(out)
        out = self.l2(out)
        out = self.a2(out)
        out = self.l3(out)
        return [out]


class Inf_High(nn.Module):
    pass


class Gen_High(nn.Module):
    pass


# class Inf_Bot(nn.Module):
# 
#     def __init__(self, batch_size, nx, nh, nz):
#         super(Inf_Bot, self).__init__()
# 
#         self.batch_size = batch_size
#         self.nz = nz
#         self.l1 = nn.Linear(nx, nh)
#         self.a1 = nn.LeakyReLU(0.01)
#         self.l2 = nn.Linear(nh,nh)
#         self.a2 = nn.LeakyReLU(0.01)
#         self.l3_mu = nn.Linear(nh, nz)
#         self.l3_sig = nn.Linear(nh, nz)
#         self.sigmoid = nn.Sigmoid()
# 
#     def forward(self, x):
#         out = self.l1(x)
#         out = self.a1(out)
#         out = self.l2(out)
#         out = self.a2(out)
#         out_mu = self.l3_mu(out)
#         out_sig = self.sigmoid(self.l3_sig(out))
# 
#         out = out_mu + out_sig * to_var(torch.randn(self.batch_size, self.nz))
# 
#         return out
# 
# 
# 
# class Inf_Bot_Conv(nn.Module):
# 
#     def __init__(self, batch_size, nz):
#         super(Inf_Bot_Conv, self).__init__()
# 
#         norm = LayerNorm1d
# 
#         self.batch_size = batch_size
# 
#         self.l1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=5, padding=2, stride=2),
#             #nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2),
#             #nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=2),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2))
# 
#         self.l2 = nn.Sequential(
#             nn.Linear(512*4*4, nz))
# 
#     def forward(self, z):
#         out = self.l1(z)
#         out = out.view(self.batch_size, -1)
#         out = self.l2(out)
#         return out
# 
# 
# 
# 
# class Inf_Bot_Conv32(nn.Module):
# 
#     def __init__(self, batch_size, nz):
#         super(Inf_Bot_Conv32, self).__init__()
# 
#         self.batch_size = batch_size
# 
#         self.l1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=5, padding=2, stride=2),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.02),
#             nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.02),
#             nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.02))
# 
#         self.l2 = nn.Sequential(
#             nn.Linear(256*4*4, nz))
# 
#     def forward(self, x):
#         out = self.l1(x)
#         out = out.view(self.batch_size, -1)
#         out = self.l2(out)
#         return out





