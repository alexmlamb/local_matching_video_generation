#!/usr/bin/env python
import sys
sys.path.insert(0, '/u/lambalex/.local/lib/python2.7/site-packages/torch-0.2.0+4af66c4-py2.7-linux-x86_64.egg')
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from torch.autograd import Variable, grad
from torchvision.utils import save_image
import os
slurm_name = os.environ["SLURM_JOB_ID"]

from reg_loss import gan_loss
from LayerNorm1d import LayerNorm1d

'''
Initially just implement LSGAN on MNIST.  

Then implement a critic.  
'''

batch_size = 128

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=True)

def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)

pacman_data = np.load('pacman_data_20k.npy')

pacman_data = pacman_data#(pacman_data + 1)/2.0

print pacman_data.dtype

#Discriminator
#D = nn.Sequential(
#    nn.Linear(32*32*3, 512),
#    nn.LeakyReLU(0.02),
#    nn.Linear(512, 512),
#    nn.LeakyReLU(0.02),
#    nn.Linear(512, 1))

from D_Bot import D_Bot_Conv32
D = D_Bot_Conv32(batch_size)

from Gen_Bot import Gen_Bot_Conv32
G = Gen_Bot_Conv32(batch_size, 64)

# Generator 
#G = nn.Sequential(
#    nn.Linear(64, 512),
#    nn.LeakyReLU(0.2),
#    nn.Linear(512, 512),
#    nn.LeakyReLU(0.2),
#    nn.Linear(512, 32*32*3),
#    nn.Tanh())


if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(0.5,0.99))
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.5,0.99))

for epoch in range(200):
    for i in range(0, 19000, batch_size):

        images = torch.from_numpy(pacman_data[0,i:i+batch_size,:,:,:])

        images = to_var(images)

        use_penalty = True
        print "up", use_penalty

        outputs = D(images)
        d_loss_real = gan_loss(pre_sig=outputs, real=True, D=True, use_penalty=use_penalty,grad_inp=images,gamma=1.0)

        real_score = outputs

        z = to_var(torch.randn(batch_size, 64))
        fake_images = G(z)
        outputs = D(fake_images)

        d_loss_fake = gan_loss(pre_sig=outputs, real=False, D=True, use_penalty=use_penalty,grad_inp=fake_images,gamma=1.0)

        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake

        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()


        print "fake scores D", fake_score.mean()
        print "real scores D", real_score.mean()
        print "epoch", epoch
        #============TRAIN GENERATOR========================$

        z = to_var(torch.randn(batch_size, 64))
        fake_images = G(z)

        outputs = D(fake_images)
     
        g_loss = gan_loss(pre_sig=outputs, real=False, D=False, use_penalty=False,grad_inp=None)

        D.zero_grad()
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()


    fake_images = fake_images.view(fake_images.size(0), 3, 32, 32)
    save_image(denorm(fake_images.data), './data/%s_fake_images.png' %(slurm_name))

    real_images = images.view(images.size(0), 3, 32, 32)
    print "real images min max", real_images.min(), real_images.max()
    save_image(denorm(real_images.data), './data/%s_real_images.png' %(slurm_name))


