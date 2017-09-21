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

batch_size = 100

real_labels = to_var(torch.ones(batch_size))
fake_labels = to_var(torch.zeros(batch_size))
boundary_labels = to_var(0.5 * torch.ones(batch_size))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])

mnist = datasets.MNIST(root='./data/', train=True, download=True, transform=transform)


data_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)


nz = 128
ns = 2

#(zL,xL) and (zR,xR)
d_bot = nn.Sequential(
    nn.Linear(784/ns + nz, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1))

#(z,zL,zR)
d_top = nn.Sequential(
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

models = [d_top, d_bot, inf_bot, gen_bot, inf_top, gen_top]

if torch.cuda.is_available():
    for model in models: 
        model.cuda()

d_top_optimizer = torch.optim.Adam(d_top.parameters(), lr=0.0003)
d_bot_optimizer = torch.optim.Adam(d_bot.parameters(), lr=0.0003)
inf_bot_optimizer = torch.optim.Adam(inf_bot.parameters(), lr=0.0003)
gen_bot_optimizer = torch.optim.Adam(gen_bot.parameters(), lr=0.0003)
inf_top_optimizer = torch.optim.Adam(inf_top.parameters(), lr=0.0003)
gen_top_optimizer = torch.optim.Adam(gen_top.parameters(), lr=0.0003)


for epoch in range(200):
    for i, (images, _) in enumerate(data_loader):

        #batch_size = images.size(0)

        #====
        #Inference Procedure

        #gen_bot, then 

        #====

        images = to_var(images.view(batch_size, -1))

        z_bot_lst = []
        for seg in range(0,ns):
            xs = images[:,seg*(784/ns):(seg+1)*(784/ns)]
            zs = inf_bot(xs)

            d_out_bot = d_bot(torch.cat((zs,xs),1))
            
            d_loss_bot = ((d_out_bot - real_labels)**2).mean()
            g_loss_bot = ((d_out_bot - boundary_labels)**2).mean()

            d_bot.zero_grad()
            d_loss_bot.backward(retain_graph=True)
            d_bot_optimizer.step()

            inf_bot.zero_grad()
            g_loss_bot.backward()
            inf_bot_optimizer.step()

            z_bot_lst.append(zs)

        z_bot = torch.cat(z_bot_lst, 1)

        z_bot = Variable(z_bot.data)

        z_top = inf_top(z_bot)

        d_out_top = d_top(torch.cat((z_top, z_bot), 1))

        d_loss_top = ((d_out_top - real_labels)**2).mean()
        g_loss_top = ((d_out_top - boundary_labels)**2).mean()

        d_top.zero_grad()
        d_loss_top.backward(retain_graph=True)
        d_top_optimizer.step()

        inf_top.zero_grad()
        g_loss_top.backward()
        inf_top_optimizer.step()


        print "epoch", epoch
        #============GENERATION PROCESS========================$

        z_top = to_var(torch.randn(batch_size, nz))

        z_bot = gen_top(z_top)

        d_out_top = d_top(torch.cat((z_top, z_bot),1))

        d_top.zero_grad()

        d_loss_top = ((d_out_top - fake_labels)**2).mean()
        d_loss_top.backward(retain_graph=True)
        d_top_optimizer.step()

        g_loss_bot = ((d_out_top - boundary_labels)**2).mean()
        gen_bot.zero_grad()
        d_bot.zero_grad()
        g_loss_bot.backward()
        gen_top_optimizer.step()

        z_bot = Variable(z_bot.data)

        gen_x_lst = []
        for seg in range(0,ns):
            seg_z = z_bot[:,seg*nz:(seg+1)*nz]
            seg_x = gen_bot(seg_z)
            gen_x_lst.append(seg_x)
            d_out_bot = d_bot(torch.cat((seg_z, seg_x),1))
            d_loss_bot = ((d_out_bot - fake_labels)**2).mean()
            
            d_bot.zero_grad()
            d_loss_bot.backward(retain_graph=True)
            d_bot_optimizer.step()

            g_loss_bot = ((d_out_bot - boundary_labels)**2).mean()
            gen_bot.zero_grad()
            d_bot.zero_grad()
            g_loss_bot.backward()
            gen_bot_optimizer.step()

        print d_out_bot


    fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images.data), './data/%s_fake_images.png' %(slurm_name))



