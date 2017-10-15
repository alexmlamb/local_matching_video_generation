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
from torch.autograd import Variable
from torch.autograd import grad
from torchvision.utils import save_image
import os
slurm_name = os.environ["SLURM_JOB_ID"]
from utils import to_var
from LayerNorm1d import LayerNorm1d
#from gradient_penalty import gradient_penalty
import random
import numpy as np
from gan_loss import ls_loss
from reg_loss import gan_loss_multi
import math

'''
Initially just implement LSGAN on MNIST.  

Then implement a critic.  
'''


def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)

def imgnorm(x):
    out = (x/255.0)*2 - 1
    return out

batch_size = 100

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])

#mnist = datasets.MNIST(root='./data/', train=True, download=True, transform=transform)

#data_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)

mnist = np.load('/u/lambalex/Downloads/mnist_test_seq.npy').astype('float32')

nz = 64
ns = 4 #20

from D_Top import D_Top
d_top = D_Top(batch_size, nz*ns, nz, 256)

from D_Bot import D_Bot
d_bot = D_Bot(batch_size, nz)
d_bot.apply(weights_init)

print "d bot", d_bot

from Inf_Bot import Inf_Bot
inf_bot = Inf_Bot(batch_size, nz)

#(zL->xL) and (zR->xR)
from Gen_Bot import Gen_Bot_Conv
gen_bot = Gen_Bot_Conv(batch_size, nz)
gen_bot.apply(weights_init)

print "gen bot", gen_bot

#(zL,zR -> z)
inf_top = nn.Sequential(
    nn.Linear(nz*ns, 256),
    nn.BatchNorm1d(256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.BatchNorm1d(256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, nz))

from Gen_Top import Gen_Top
gen_top = Gen_Top(batch_size, nz, 256, ns*nz)

models = [d_top, d_bot, inf_bot, gen_bot, inf_top, gen_top]

if torch.cuda.is_available():
    for model in models: 
        model.cuda()

lr = 0.00001

print "lr", lr

d_top_optimizer = torch.optim.RMSprop(d_top.parameters(), lr=0.0001)
d_bot_optimizer = torch.optim.RMSprop(d_bot.parameters(), lr=lr)
inf_bot_optimizer = torch.optim.RMSprop(inf_bot.parameters(), lr=0.0001)
gen_bot_optimizer = torch.optim.RMSprop(gen_bot.parameters(), lr=lr)
inf_top_optimizer = torch.optim.RMSprop(inf_top.parameters(), lr=0.0001)
gen_top_optimizer = torch.optim.RMSprop(gen_top.parameters(), lr=0.0001)


for epoch in range(5000):
    for i in range(0,1000-batch_size,batch_size):

        images = imgnorm(torch.from_numpy(mnist[:,i:i+batch_size,:,:]))

        #============GENERATION PROCESS========================$

        z_top = to_var(torch.randn(batch_size, nz))

        z_bot = gen_top(z_top)

        d_out_top = d_top(z_bot)

        d_top.zero_grad()

        d_loss_top = ls_loss(d_out_top, 0).mean()

        d_loss_top.backward(retain_graph=True)
        d_top_optimizer.step()

        g_loss_top = ls_loss(d_out_top, 1.0).mean()

        gen_top.zero_grad()
        d_top.zero_grad()
        g_loss_top.backward()
        gen_top_optimizer.step()

        z_bot = Variable(z_bot.data)

        gen_x_lst = []


        #====
        #Inference Procedure
        #====

        images = to_var(images)
        real_images = images

        z_bot_lst = []
        for seg in range(0,ns):

            #====================================Low level inference and reconstruction=======================

            xs = images[seg]#images[:,seg*(784/ns):(seg+1)*(784/ns)]
            
            xs = xs.view(100,1,64,64)
            zs = inf_bot(xs)
            reconstruction = gen_bot(zs)

            d_out_bot = d_bot(xs, zs)

            g_loss_bot = gan_loss_multi(pre_sig_lst=d_out_bot, real=True, D=False, use_penalty=False,grad_inp=None)

            rec_loss = ((reconstruction - xs)**2).mean()

            if False:
                g_loss_bot += 0.1 * rec_loss
                print "training with reconstruction loss"
            else:
                print "training without low level reconstruction loss"

            gen_bot.zero_grad()
            inf_bot.zero_grad()
            d_bot.zero_grad()
            g_loss_bot.backward(retain_graph=True)
            gen_bot_optimizer.step()
            inf_bot_optimizer.step()

            xsi = Variable(xs.data, requires_grad=True)
            zsi = Variable(zs.data, requires_grad=True)

            d_out_bot = d_bot(xsi, zsi)

            #d_loss_bot = reg_loss(d_out_bot, "d", "r", xsi)
            d_loss_bot = gan_loss_multi(pre_sig_lst=d_out_bot, real=True, D=True, use_penalty=True,grad_inp=xsi,gamma=1.0)

            d_bot.zero_grad()
            d_loss_bot.backward()
            d_bot_optimizer.step()

            z_bot_lst.append(zs)


            #============================Low level generation==================================================

            seg_z = z_bot[:,seg*nz:(seg+1)*nz] * 1.0 + 0.0 * to_var(torch.randn(batch_size, nz))
            seg_x = gen_bot(seg_z)
            gen_x_lst.append(seg_x)

            d_out_bot = d_bot(seg_x, seg_z)

            g_loss_bot = ls_loss(d_out_bot, 0.5)
            #g_loss_bot = reg_loss(d_out_bot, "g", "f")

            gen_bot.zero_grad()
            d_bot.zero_grad()
            g_loss_bot.backward(retain_graph=True)
            gen_bot_optimizer.step()

            xsi = Variable(seg_x.data, requires_grad=True)
            zsi = Variable(seg_z.data, requires_grad=True)

            d_out_bot = d_bot(xsi,zsi)

            #d_loss_bot = ls_loss(d_out_bot, 0.0)
            d_loss_bot = gan_loss_multi(pre_sig_lst=d_out_bot, real=False, D=True, use_penalty=True,grad_inp=xsi,gamma=1.0)
            #d_loss_bot = reg_loss(d_out_bot, "d", "f", xsi)

            d_bot.zero_grad()
            d_loss_bot.backward()
            d_bot_optimizer.step()

        z_bot = torch.cat(z_bot_lst, 1)

        z_bot = Variable(z_bot.data)

        z_top = inf_top(z_bot)

        z_bot_rec = gen_top(z_top)

        reconstruction_loss = ((z_bot - z_bot_rec)**2).mean()

        #print "high level rec loss", reconstruction_loss

        d_out_top = d_top(z_bot)

        d_loss_top = ls_loss(d_out_top, 1).mean()#((d_out_top - real_labels)**2).mean()
        g_loss_top = ls_loss(d_out_top, 0.5).mean()#((d_out_top - boundary_labels)**2).mean()

        #print "d loss top inf", d_loss_top

        #d_loss_top += 0.1 * gradient_penalty(d_out_top.norm(2), z_bot)

        if True:
            print "optimizing for high level rec loss"
            g_loss_top += 0.1 * reconstruction_loss
        else:
            print "not optimizing high level rec loss"

        d_top.zero_grad()
        d_loss_top.backward(retain_graph=True)
        d_top_optimizer.step()

        inf_top.zero_grad()
        gen_top.zero_grad()
        d_top.zero_grad()

        g_loss_top.backward()
        inf_top_optimizer.step()
        gen_top_optimizer.step()

        print "epoch", epoch


    print "len genx list", len(gen_x_lst)

    for seg in range(0, ns):
        fake_images = gen_x_lst[seg]
        #print "fake images min max", fake_images.min(), fake_images.max()
        print "fake images shape", fake_images.size()
        save_image(denorm(fake_images.data), './data/%s_fake_images_%i.png' %(slurm_name, seg))

    
    for seg in range(0, ns):
        save_image(denorm(real_images[seg].view(100,1,64,64).data), './data/%s_real_images_%i.png' %(slurm_name, seg))

    
    #z_bot_lst = []
    x_bot_lst = []
    z_bot_lst = []
    for seg in range(0,ns):
        xs = real_images[seg].view(100,1,64,64)
        zs = inf_bot(xs)
        xr = gen_bot(zs)
        x_bot_lst.append(xr)
        z_bot_lst.append(zs)

        save_image(denorm(xr.data), './data/%s_rec_images_bot_%i.png' %(slurm_name, seg))

    z_top = to_var(torch.randn(batch_size, nz))
    z_bot = gen_top(z_top)

    for seg in range(0,ns):
        seg_z = z_bot[:,seg*nz:(seg+1)*nz]
        xg = gen_bot(seg_z)

        save_image(denorm(xg.data), './data/%s_sample_gen_%i.png' %(slurm_name, seg))





