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
from utils import to_var, make_dir_if_not_exists
from LayerNorm1d import LayerNorm1d
from gradient_penalty import gradient_penalty
import random
from timeit import default_timer as timer
import pickle
from math import sqrt
from reg_loss import gan_loss

'''
Initially just implement LSGAN on MNIST.  

Then implement a critic.  
'''

slurm_name = os.environ["SLURM_JOB_ID"]
DATASET = 'lsun_bedroom'
DATA_DIR = os.path.join(os.path.abspath('data'), DATASET)
# OUT_DIR = os.path.join('/scratch/nealbray/loc', DATASET, slurm_name)
OUT_DIR = os.path.join('/data/lisatmp4/nealbray/loc', DATASET, slurm_name)
MODELS_DIR = os.path.join(OUT_DIR, 'saved_models')
SUM_DISC_OUTS = False
Z_NORM_MULT = 1e-3
Z_NORM_MULT = None
CHECKPOINT_INTERVAL = 1 * 60
LOWER_ONLY = False
REC_PENALTY = False
REC_SHORTCUT = True
HIGH_SHORTCUT = True
LOAD_LOWER = True
HIGHER_ONLY = True
if HIGHER_ONLY:
    LOAD_LOWER = True
LOWER_SLURM_ID = 65961
SAVED_MODELS_DIR = '/data/lisatmp4/nealbray/loc/lsun_bedroom/%d/saved_models' % LOWER_SLURM_ID

start_time = timer()


def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)


def torch_to_norm(zs):
    return zs.norm(2).data.cpu().numpy()[0]


batch_size = 32
IMAGE_LENGTH = 64
NUM_CHANNELS = 3
print 'Loading LSUN bedrooms dataset'
# dataset = datasets.LSUN('/scratch/nealbray/lsun', classes=['bedroom_train'],
dataset = datasets.LSUN('/data/lisa/data/lsun', classes=['bedroom_train'],
                        transform=transforms.Compose([
                        transforms.Scale(IMAGE_LENGTH),
                        transforms.CenterCrop(IMAGE_LENGTH),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]))
nz = 64
ns = 16
nz_high = 256
low_z_dim = 4*4*32
total_low_z =low_z_dim*ns

NUM_DISC_OUTPUTS = 4
real_labels = to_var(torch.ones(batch_size, NUM_DISC_OUTPUTS))
fake_labels = to_var(torch.zeros(batch_size, NUM_DISC_OUTPUTS))
boundary_labels = to_var(0.5 * torch.ones(batch_size, NUM_DISC_OUTPUTS))


data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
ns_per_dim = int(sqrt(ns))

seg_length = IMAGE_LENGTH / ns_per_dim

print "ns", ns

d_bot = torch.load(os.path.join(SAVED_MODELS_DIR, '%d_dbot.pt' % LOWER_SLURM_ID))
inf_bot = torch.load(os.path.join(SAVED_MODELS_DIR, '%d_infbot.pt' % LOWER_SLURM_ID))
gen_bot = torch.load(os.path.join(SAVED_MODELS_DIR, '%d_genbot.pt' % LOWER_SLURM_ID))

# from D_Top import D_Top
# d_top = D_Top(batch_size, nz*ns, nz, 256)
from archs.lsun import Disc_High
d_top = nn.Sequential(
    nn.Linear(total_low_z, 2048),
    #nn.BatchNorm1d(2048),
    nn.LeakyReLU(0.02),
    nn.Linear(2048, 1024),
    #nn.BatchNorm1d(1024),
    nn.LeakyReLU(0.02),
    nn.Linear(1024, 1))

#(zL,zR -> z)
inf_top = nn.Sequential(
    nn.Linear(nz*ns, 256),
    nn.BatchNorm1d(256),
    nn.LeakyReLU(0.02),
    nn.Linear(256, 256),
    nn.BatchNorm1d(256),
    nn.LeakyReLU(0.02),
    nn.Linear(256, nz))

from archs.lsun import Gen_High
gen_top = Gen_High(batch_size, nz_high, total_low_z)

models = [d_top, d_bot, inf_bot, gen_bot, inf_top, gen_top]

if torch.cuda.is_available():
    for model in models: 
        model.cuda()

d_top_optimizer = torch.optim.Adam(d_top.parameters(), lr=0.0003, betas=(0.5,0.99))
inf_top_optimizer = torch.optim.Adam(inf_top.parameters(), lr=0.0003, betas=(0.5,0.99))
gen_top_optimizer = torch.optim.Adam(gen_top.parameters(), lr=0.0003, betas=(0.5,0.99))

z_bot_norms = []
z_top_norms = []
checkpoint_i = 1
for epoch in range(200):
    for i, (images, _) in enumerate(data_loader):

        #====
        #Inference Procedure
        #====

        print '#####size:', to_var(images).size()
        # images = to_var(images.view(batch_size, -1))
        images = to_var(images)

        print "images min max", images.min(), images.max()

        z_bot_lst = []
        for seg in range(0, ns):
            # Infer lower level z from data
            i = seg / ns_per_dim
            j = seg % ns_per_dim
            xs = images[:, :, i*seg_length:(i+1)*seg_length, j*seg_length:(j+1)*seg_length]
            z_volume = inf_bot(xs, take_pre=True)
            zs = z_volume.view(batch_size, -1)
            z_bot_lst.append(zs)

        z_bot = torch.cat(z_bot_lst, 1)
        z_bot = to_var(z_bot.data)


        #### Higher Level Inference
        # z_top = inf_top(z_bot)
        # 
        # # Reconstruct lower level z through higher level z
        # # Currently used for higher level generator learning
        # z_bot_rec = gen_top(z_top)
        # reconstruction_loss = ((z_bot - z_bot_rec)**2).mean()
        
        # print "high level rec loss", reconstruction_loss
        
        # Discriminator on only lower z (not ALI)
        d_out_top = d_top(z_bot)
        
        d_loss_top = gan_loss(pre_sig=d_out_top, real=True, D=True, use_penalty=True, grad_inp=z_bot, gamma=1.0)
        g_loss_top = gan_loss(pre_sig=d_out_top, real=True, D=False, use_penalty=False, grad_inp=None, gamma=None, bgan=True)
        # d_loss_top = ((d_out_top - real_labels)**2).mean()
        # g_loss_top = ((d_out_top - boundary_labels)**2).mean()
        
        print "d loss top inf", d_loss_top

        if REC_PENALTY:
            print "optimizing for high level rec loss"
            g_loss_top += 10.0 * reconstruction_loss
        else:
            print "not optimizing high level rec loss"
        
        d_top.zero_grad()
        d_loss_top.backward(retain_graph=True)
        d_top_optimizer.step()
        
        # inf_top.zero_grad()
        gen_top.zero_grad()
        d_top.zero_grad()
        g_loss_top.backward(retain_graph=True)
        # inf_top_optimizer.step()
        gen_top_optimizer.step()

        #============GENERATION PROCESS========================$

        # Sample higher and lower z
        z_top = to_var(torch.randn(batch_size, nz_high))
        z_bot = gen_top(z_top)
        
        d_out_top = d_top(z_bot)
        
        d_top.zero_grad()
        
        d_loss_top = gan_loss(pre_sig=d_out_top, real=False, D=True, use_penalty=True, grad_inp=z_top, gamma=1.0)
        # d_loss_top += ((d_out_top - fake_labels)**2).mean()
        
        d_loss_top.backward(retain_graph=True)
        d_top_optimizer.step()
        
        print "d loss top gen", d_loss_top
        
        g_loss_top = gan_loss(pre_sig=d_out_top, real=False, D=False, use_penalty=False, grad_inp=None, gamma=None, bgan=True)
        # g_loss_top = ((d_out_top - boundary_labels)**2).mean()
        
        gen_top.zero_grad()
        d_top.zero_grad()
        g_loss_top.backward(retain_graph=True)
        gen_top_optimizer.step()
        
        z_bot = to_var(z_bot.data)

        gen_x_lst = []
        for seg in range(0,ns):
            seg_z = z_bot[:,seg*low_z_dim:(seg+1)*low_z_dim].contiguous()
            z_volume = seg_z.view(batch_size, 32, 4, 4)
            seg_x = gen_bot(z_volume, give_pre=True)
            
            gen_x_lst.append(seg_x)

        #print d_out_bot
        elapsed = timer() - start_time
        if elapsed > checkpoint_i * CHECKPOINT_INTERVAL:
            print 'Writing images and checkpoints'
            make_dir_if_not_exists(OUT_DIR)
            make_dir_if_not_exists(MODELS_DIR)
        
            # Log z norms
            z_bot_norm = map(torch_to_norm, z_bot_lst)
            z_bot_norms.append(max(z_bot_norm))
            d = {'z_bot_norms': z_bot_norms}
            if not LOWER_ONLY:
                z_top_norm = torch_to_norm(z_top)
                z_top_norms.append(z_top_norm)
                d['z_top_norms'] = z_top_norms
            with open(os.path.join(OUT_DIR, 'z_norms.pkl'), 'wb') as f:
                pickle.dump(d, f)
        
            # fake_images = torch.cat(gen_x_lst, 1)
            # fake_images = fake_images.view(fake_images.size(0), NUM_CHANNELS, IMAGE_LENGTH, IMAGE_LENGTH)
            
            fake_images = torch.zeros(batch_size, NUM_CHANNELS, IMAGE_LENGTH, IMAGE_LENGTH)
            for seg in range(0, ns):
                x_seg = gen_x_lst[seg]
                i = seg / ns_per_dim
                j = seg % ns_per_dim
                fake_images[:, :, i*seg_length:(i+1)*seg_length, j*seg_length:(j+1)*seg_length] = x_seg.data
        
            save_image(denorm(fake_images), os.path.join(OUT_DIR, 'fake_images%05d.png' % checkpoint_i))
        
            real_images = images.view(images.size(0), NUM_CHANNELS, IMAGE_LENGTH, IMAGE_LENGTH)
            save_image(denorm(real_images.data), os.path.join(OUT_DIR, 'real_images%05d.png' % checkpoint_i))

            # x_bot_lst = []
            # z_bot_lst = []
            rec_images_bot = torch.zeros(batch_size, NUM_CHANNELS, IMAGE_LENGTH, IMAGE_LENGTH)
            for seg in range(0,ns):
                i = seg / ns_per_dim
                j = seg % ns_per_dim
                xs = images[:, :, i*seg_length:(i+1)*seg_length, j*seg_length:(j+1)*seg_length]
                if REC_SHORTCUT:
                    xr = gen_bot(inf_bot(xs, take_pre=True), give_pre=True)
                else:
                    xr = gen_bot(inf_bot(xs))
                rec_images_bot[:, :, i*seg_length:(i+1)*seg_length, j*seg_length:(j+1)*seg_length] = xr.data
                # x_bot_lst.append(xr)
                # z_bot_lst.append(zs)

            # rec_images_bot = torch.cat(x_bot_lst, 1)
            # 
            # rec_images_bot = rec_images_bot.view(rec_images_bot.size(0), NUM_CHANNELS, IMAGE_LENGTH, IMAGE_LENGTH)
            save_image(denorm(rec_images_bot), os.path.join(OUT_DIR, 'rec_images_bot%05d.png' % checkpoint_i))
        
            if not LOWER_ONLY:
                # z_bot = torch.cat(z_bot_lst, 1)
                # z_top = inf_top(z_bot)
                # z_bot = gen_top(z_top)
                # 
                # gen_x_lst = []
                # for seg in range(0,ns):
                #     seg_z = z_bot[:,seg*low_z_dim:(seg+1)*low_z_dim].contiguous()
                #     z_volume = seg_z.view(batch_size, 32, 4, 4)
                #     seg_x = gen_bot(z_volume, give_pre=True)
                #     gen_x_lst.append(seg_x)
            
                rec_images_top = torch.cat(gen_x_lst, 1)
            
                rec_images_top = rec_images_top.view(rec_images_top.size(0), NUM_CHANNELS, IMAGE_LENGTH, IMAGE_LENGTH)
                save_image(denorm(rec_images_top.data), os.path.join(OUT_DIR, 'rec_images_top%05d.png' % checkpoint_i))
            
            # Checkpoint
            torch.save(gen_top, os.path.join(MODELS_DIR, '%s_gentop.pt' % slurm_name))
            torch.save(d_top, os.path.join(MODELS_DIR, '%s_dtop.pt' % slurm_name))
            torch.save(inf_top, os.path.join(MODELS_DIR, '%s_inftop.pt' % slurm_name))
                
            checkpoint_i += 1
                

end_time = timer()
elapsed = end_time - start_time
print 'total time used (in seconds):', elapsed
print 'total time used (in minutes):', elapsed / 60.0
print 'total time used (in hours):', elapsed / 60.0 / 60.0
