#!/usr/bin/env python
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable, grad
from torchvision.utils import save_image
import os
slurm_name = os.environ["SLURM_JOB_ID"]

'''
Initially just implement LSGAN on MNIST.  

Then implement a critic.  
'''

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)

real_labels = to_var(torch.ones(batch_size))
fake_labels = to_var(torch.zeros(batch_size))
boundary_labels = to_var(0.5 * torch.ones(batch_size))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])

mnist = datasets.MNIST(root='./data/', train=True, download=True, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=100, shuffle=True)

nz = 64
ns = 2

#(zL,xL) and (zR,xR)
D_bot = nn.Sequential(
    nn.Linear(784/ns + nz, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1))

#(z,zL,zR)
D_top = nn.Sequential(
    nn.Linear(nz*ns + nz, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1))

#(xL->zL) and (xR->zR)
inf_bot = nn.Sequential(
    nn.Linear(784/ns, 256),
    nn.BatchNorm1d(256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.BatchNorm1d(256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, nz),
    nn.Tanh())

#(zL->xL) and (zR->xR)
gen_bot = nn.Sequential(
    nn.Linear(nz, 256),
    nn.BatchNorm1d(256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.BatchNorm1d(256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 784/ns),
    nn.Tanh())

#(zL,zR -> z)
inf_top = nn.Sequential(
    nn.Linear(nz*ns, 256),
    nn.BatchNorm1d(256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.BatchNorm1d(256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, nz),
    nn.Tanh())

#(z -> zL,zR)
gen_top = nn.Sequential(
    nn.Linear(nz, 256),
    nn.BatchNorm1d(256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.BatchNorm1d(256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, ns*nz),
    nn.Tanh())

models = [D_top, D_bot, inf_bot, gen_bot, inf_top, gen_top]

if torch.cuda.is_available():
    for model in models: 
        model.cuda()

d_top_optimizer = torch.optim.Adam(D_top.parameters(), lr=0.0003)
d_bot_optimizer = torch.optim.Adam(D_bot.parameters(), lr=0.0003)
inf_bot_optimizer = torch.optim.Adam(inf_bot.parameters(), lr=0.0003)
gen_bot_optimizer = torch.optim.Adam(gen_bot.parameters(), lr=0.0003)
inf_top_optimizer = torch.optim.Adam(inf_top.parameters(), lr=0.0003)
gen_top_optimizer = torch.optim.Adam(gen_top.parameters(), lr=0.0003)


for epoch in range(200):
    for i, (images, _) in enumerate(data_loader):

        batch_size = images.size(0)

        #====
        #Inference Procedure
        #====

        images = to_var(images.view(batch_size, -1))


        outputs_left = D(images[:,:784/2])
        d_loss_real_left = ((outputs_left - real_labels)**2).mean()
        real_score = outputs_left

        z = to_var(torch.randn(batch_size, 64))
        fake_images = G(z)
        outputs_left = D(fake_images)
        outputs_right = 
        d_loss_fake = ((outputs - fake_labels)**2).mean()
        fake_score = outputs

        #Feed inference to discriminators

        d_loss_left = d_loss_real_left + d_loss_fake_left
        d_loss_right = d_loss_real_right + d_loss_fake_right

        D_left.zero_grad()
        D_right.zero_grad()
        d_loss.backward()
        d_optimizer.step()


        print "epoch", epoch
        #============GENERATION PROCESS========================$

        z_top = to_var(torch.randn(batch_size, 64))

        z_bot = gen_top(z_top)

        gen_x_lst = []

        for seg in range(0,ns):
            seg_z = z_bot[:,seg*nz:(seg+1)*nz]
            seg_x = gen_bot(seg_z)
            gen_x_lst.append(seg_x)
            d_out_bot = d_bot(join_bot_d(seg_x, seg_z))
            d_loss_bot = ((d_out_bot - fake_labels)**2).mean()
            
            d_bot.zero_grad()
            d_loss_bot.backward(retain_graph=True)
            d_bot_optimizer().step()

            g_loss_bot = ((d_out_bot - boundary_labels)**2).mean()
            gen_bot.zero_grad()
            d_bot.zero_grad()
            g_loss_bot.backward()
            g_bot_optimizer.step()



    fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images.data), './data/%s_fake_images.png' %(slurm_name))



