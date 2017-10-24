#!/usr/bin/env python

import sys
sys.path.insert(0, '/u/lambalex/.local/lib/python2.7/site-packages/torch-0.2.0+4af66c4-py2.7-linux-x86_64.egg')
import torch
import numpy as np
from torch.autograd import Variable, grad
from utils import to_var, denorm
from torchvision.utils import save_image
import torch.nn as nn
from reg_loss import gan_loss
import os
slurm_name = os.environ["SLURM_JOB_ID"]

'''
Trains a higher level model in isolation using a pre-trained generator and inference network from the lower level.  


'''

GB = torch.load('saved_models/65157_Gbot.pt')
IB = torch.load('saved_models/65157_Ibot.pt')

print "GB", GB

IB = IB.cuda()
GB = GB.cuda()

pacman_data = np.load('pacman_data_20k.npy')


# Generator
#Gh = nn.Sequential(
#    nn.Linear(64, 1024),
#    nn.BatchNorm1d(1024),
#    nn.LeakyReLU(0.02),
#    nn.Linear(1024, 1024),
#    nn.BatchNorm1d(1024),
#    nn.LeakyReLU(0.02),
#    nn.Linear(1024, 4*4*512*5))

from Gen_Top import Gen_Top_fc
Gh = Gen_Top_fc(128,128,4*4*32*5)

#from Gen_Top import Gen_Top_4
#Gh = Gen_Top_4(128,64)

#Discriminator
Dh = nn.Sequential(
    nn.Linear(4*4*32*5, 1024),
    #nn.BatchNorm1d(512),
    nn.LeakyReLU(0.02),
    nn.Linear(1024, 1024),
    #nn.BatchNorm1d(512),
    nn.LeakyReLU(0.02),
    nn.Linear(1024, 1))

if torch.cuda.is_available():
    Dh = Dh.cuda()
    Gh = Gh.cuda()
else:
    raise Exception("cuda not available")

dh_optimizer = torch.optim.Adam(Dh.parameters(), lr=0.0001, betas=(0.5,0.99))
gh_optimizer = torch.optim.Adam(Gh.parameters(), lr=0.0001, betas=(0.5,0.99))

for epoch in range(0,1000):

    print "epoch", epoch

    for i in range(0,19000,128):

        pacman_frame_lst = []

        for t in range(0,5):
            pacman_frames = to_var(torch.from_numpy(pacman_data[t,i:i+128,:,:,:]))
            enc = IB(pacman_frames, take_pre=True).view(128,-1)
            pacman_frame_lst.append(enc)

        real = to_var(torch.cat(pacman_frame_lst,1).data)


        real_score = Dh(real)

        d_loss_real = gan_loss(pre_sig=real_score, real=True, D=True, use_penalty=True,grad_inp=real,gamma=1.0)

        Dh.zero_grad()
        d_loss_real.backward()

        dh_optimizer.step()


        #GENERATION ===========================
        z_raw = to_var(torch.randn(128,128))

        gen_val = Gh(z_raw)

        print gen_val.size()

        fake_score = Dh(gen_val)

        d_loss_fake = gan_loss(pre_sig=fake_score, real=False, D=True, use_penalty=True,grad_inp=gen_val,gamma=1.0)

        Dh.zero_grad()
        d_loss_fake.backward(retain_graph=True)
        dh_optimizer.step()
    
        g_loss_fake = gan_loss(pre_sig=fake_score, real=False, D=False, use_penalty=False,grad_inp=None,gamma=None,bgan=True)

        Dh.zero_grad()
        Gh.zero_grad()
        g_loss_fake.backward()
        gh_optimizer.step()

    print "fake score", fake_score.mean()
    print "real score", real_score.mean()
    
    for t in range(0,5):
        one_step = gen_val[:,(32*4*4)*t:(32*4*4)*(t+1)]
        one_step = one_step.contiguous().view(128,32,4,4)
        img = GB(one_step,give_pre=True)

        save_image(denorm(img.data), 'data/%s_fake_%d.png' % (slurm_name,t))

        one_step_real = real[:,(32*4*4)*t:(32*4*4)*(t+1)]
        one_step_real = one_step_real.contiguous().view(128,32,4,4)
        img = GB(one_step_real,give_pre=True)

        save_image(denorm(img.data), 'data/%s_realrec_%d.png' % (slurm_name,t))


#rdir = 0.01 * to_var(torch.randn(128,64))

#pacman_frames = to_var(torch.from_numpy(pacman_data[t,0:0+128,:,:,:]))

#new_dec = pacman_frames

#for i in range(0,10):
#    enc = IB(new_dec)

#    new_enc = enc + rdir

#    new_dec = GB(new_enc)

#    save_image(denorm(new_dec.data), 'derp__%d.png' % i)



