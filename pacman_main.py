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
import random

from reg_loss import gan_loss
from LayerNorm1d import LayerNorm1d

load_D = True
load_G = True
load_I = True

file_D = "saved_models/62758_Dbot.pt"
file_G = "saved_models/62758_Gbot.pt"
file_I = "saved_models/62758_Ibot.pt"

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


print pacman_data.dtype

#Discriminator
#D = nn.Sequential(
#    nn.Linear(32*32*3, 512),
#    nn.LeakyReLU(0.02),
#    nn.Linear(512, 512),
#    nn.LeakyReLU(0.02),
#    nn.Linear(512, 1))

if load_D:
    print "loading from", file_D
    D = torch.load(file_D)
else:
    from D_Bot import D_Bot_Conv32
    D = D_Bot_Conv32(batch_size, 64)

if load_G:
    print "loading from", file_G
    G = torch.load(file_G)
else:
    from Gen_Bot import Gen_Bot_Conv32
    G = Gen_Bot_Conv32(batch_size, 64)

if load_I:
    print "loading from", file_I
    I = torch.load(file_I)
else:
    from Inf_Bot import Inf_Bot_Conv32
    I = Inf_Bot_Conv32(batch_size, 64)

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
    I = I.cuda()


d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(0.5,0.99))
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.5,0.99))
i_optimizer = torch.optim.Adam(I.parameters(), lr=0.0001, betas=(0.5,0.99))

print "actually running over ts!"

for epoch in range(2000):
    for i in range(0, 19000, batch_size):

        #=============TRAIN INFERENCE=========================

        time_step = random.randint(0,4)

        images = torch.from_numpy(pacman_data[time_step,i:i+batch_size,:,:,:])

        images = to_var(images)

        use_penalty = True
        print "up", use_penalty

        inf = I(images)

        print "inference mean and stdv", inf.mean(), inf.std()

        outputs = D(images, inf)

        d_loss_real = gan_loss(pre_sig=outputs, real=True, D=True, use_penalty=use_penalty,grad_inp=images,gamma=1.0) + gan_loss(pre_sig=outputs, real=True, D=True, use_penalty=use_penalty,grad_inp=inf,gamma=1.0)

        real_score = outputs

        #================GENERATOR PHASE=========================

        z = to_var(torch.randn(batch_size, 64))
        fake_images = G(z)
        outputs = D(fake_images, z)

        d_loss_fake = gan_loss(pre_sig=outputs, real=False, D=True, use_penalty=use_penalty,grad_inp=fake_images,gamma=1.0) + gan_loss(pre_sig=outputs, real=False, D=True, use_penalty=use_penalty,grad_inp=z,gamma=1.0)

        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake

        D.zero_grad()
        d_loss.backward(retain_graph=True)
        d_optimizer.step()

        print "fake scores D", fake_score.mean()
        print "real scores D", real_score.mean()
        print "epoch", epoch
     
        g_loss = gan_loss(pre_sig=fake_score, real=False, D=False, use_penalty=False,grad_inp=None)

        D.zero_grad()
        G.zero_grad()
        I.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        i_optimizer.step()

        inf_short = I(images, take_pre=True)
        rec = G(inf_short, give_pre=True)

        rec_loss = 100.0 * ((rec - images)**2).mean()

        i_loss = gan_loss(pre_sig=real_score, real=True, D=False, use_penalty=False,grad_inp=None) + rec_loss
        D.zero_grad()
        G.zero_grad()
        I.zero_grad()
        i_loss.backward()
        g_optimizer.step()
        i_optimizer.step()

    if epoch % 10 == 0:
        print "saving model"

        torch.save(G, 'saved_models/%s_Gbot.pt' %(slurm_name))
        torch.save(D, 'saved_models/%s_Dbot.pt' %(slurm_name))
        torch.save(I, 'saved_models/%s_Ibot.pt' %(slurm_name))

    fake_images = fake_images.view(fake_images.size(0), 3, 32, 32)
    save_image(denorm(fake_images.data), './data/%s_fake_images.png' %(slurm_name))

    inf_short = I(images, take_pre=True)
    rec_images = G(inf_short, give_pre=True)
    save_image(denorm(rec_images.data), './data/%s_rec_bot_images.png' %(slurm_name))

    real_images = images.view(images.size(0), 3, 32, 32)
    print "real images min max", real_images.min(), real_images.max()
    save_image(denorm(real_images.data), './data/%s_real_images.png' %(slurm_name))

    


