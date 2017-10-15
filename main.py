#!/usr/bin/env python
import sys
sys.path.insert(0, '/u/lambalex/.local/lib/python2.7/site-packages/torch-0.2.0+4af66c4-py2.7-linux-x86_64.egg')
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

from reg_loss import gan_loss
from LayerNorm1d import LayerNorm1d

'''
Initially just implement LSGAN on MNIST.  

Then implement a critic.  
'''

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=True)

def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])

mnist = datasets.MNIST(root='./data/', train=True, download=True, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=100, shuffle=True)


#Discriminator
D = nn.Sequential(
    nn.Linear(784, 1024),
    nn.LeakyReLU(0.2),
    nn.Linear(1024, 1024),
    #nn.BatchNorm1d(256),
    nn.LeakyReLU(0.2),
    nn.Linear(1024, 1024),
    
    #nn.LeakyReLU(0.2),
    #nn.Linear(1024, 1024),
    
    #nn.BatchNorm1d(256),
    nn.LeakyReLU(0.2),
    nn.Linear(1024, 1))

# Generator 
G = nn.Sequential(
    nn.Linear(64, 1024),
    #nn.BatchNorm1d(256),
    nn.LeakyReLU(0.2),
    nn.Linear(1024, 1024),
    
    LayerNorm1d(1024),

    nn.LeakyReLU(0.2),
    nn.Linear(1024, 784),
    nn.Tanh())

if torch.cuda.is_available():
    D.cuda()
    G.cuda()

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)

for epoch in range(200):
    for i, (images, _) in enumerate(data_loader):

        batch_size = images.size(0)

        images = to_var(images.view(batch_size, -1))

        real_labels = to_var(torch.ones(batch_size))
        fake_labels = to_var(torch.zeros(batch_size))
        boundary_labels = to_var(0.5 * torch.ones(batch_size))

        use_penalty = True
        print "up", use_penalty

        outputs = D(images)
        #d_loss_real = ((outputs - real_labels)**2).mean()
        d_loss_real = gan_loss(pre_sig=outputs, real=True, D=True, use_penalty=use_penalty,grad_inp=images,gamma=1.0)

        real_score = outputs

        z = to_var(torch.randn(batch_size, 64))
        fake_images = G(z)
        outputs = D(fake_images)

        #d_loss_fake = ((outputs - fake_labels)**2).mean()
        d_loss_fake = gan_loss(pre_sig=outputs, real=False, D=True, use_penalty=use_penalty,grad_inp=fake_images,gamma=1.0)

        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake

        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()


        print "fake scores D", fake_score[0:10]
        print "real scores D", real_score[0:10]
        print "epoch", epoch
        #============TRAIN GENERATOR========================$

        z = to_var(torch.randn(batch_size, 64))
        fake_images = G(z)

        outputs = D(fake_images)
        #g_loss = ((outputs - boundary_labels)**2).mean()
     
        g_loss = gan_loss(pre_sig=outputs, real=False, D=False, use_penalty=False,grad_inp=None)

        D.zero_grad()
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images.data), './data/%s_fake_images.png' %(slurm_name))



