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
DATASET = 'cifar'
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
LOWER_SLURM_ID = '74342'
LOWER_SLURM_FOLDER = '65961_lower'
SAVED_MODELS_DIR = os.path.join('/data/lisatmp4/nealbray/loc', DATASET, LOWER_SLURM_ID, 'saved_models')

start_time = timer()


def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)


def torch_to_norm(zs):
    return zs.norm(2).data.cpu().numpy()[0]


if DATASET == 'lsun_bedroom':
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
elif DATASET == 'cifar':
    batch_size = 64
    IMAGE_LENGTH = 32
    NUM_CHANNELS = 3
    dataset = datasets.CIFAR10('/data/lisa/data/cifar10', train=True, download=False,
                        transform=transforms.Compose([
                        transforms.CenterCrop(IMAGE_LENGTH),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]))
    nz = 64
    ns = 4
    nz_high = 512
else:
    raise ValueError('Unsupported dataset: %s' % DATASET)

data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

low_z_dim = 32*4*4
total_low_z =low_z_dim*ns
ns_per_dim = int(sqrt(ns))
seg_length = IMAGE_LENGTH / ns_per_dim

d_bot = torch.load(os.path.join(SAVED_MODELS_DIR, '%s_dbot.pt' % LOWER_SLURM_ID))
inf_bot = torch.load(os.path.join(SAVED_MODELS_DIR, '%s_infbot.pt' % LOWER_SLURM_ID))
gen_bot = torch.load(os.path.join(SAVED_MODELS_DIR, '%s_genbot.pt' % LOWER_SLURM_ID))

from archs.cifar import Disc_High
d_top = Disc_High(batch_size)
# from archs.cifar import Disc_High_fc
# d_top = Disc_High_fc(batch_size, total_low_z)

# from archs.cifar import Inf_High
# inf_top = Inf_High(batch_size, nz_high)
from archs.cifar import Inf_High_fc
inf_top = Inf_High_fc(batch_size, total_low_z, nz_high)

# from archs.cifar import Gen_High
# gen_top = Gen_High(batch_size, nz_high)
from archs.cifar import Gen_High_fc
gen_top = Gen_High_fc(batch_size, nz_high, total_low_z)

models = [d_top, d_bot, inf_bot, gen_bot, inf_top, gen_top]

if torch.cuda.is_available():
    for model in models:
        model.cuda()

d_top_optimizer = torch.optim.Adam(d_top.parameters(), lr=0.0001, betas=(0.5,0.99))
inf_top_optimizer = torch.optim.Adam(inf_top.parameters(), lr=0.0001, betas=(0.5,0.99))
gen_top_optimizer = torch.optim.Adam(gen_top.parameters(), lr=0.0001, betas=(0.5,0.99))

checkpoint_i = 1
for epoch in range(200):
    for i, (images, _) in enumerate(data_loader):
        
        if images.size(0) != batch_size:
            continue

        #====
        #Inference Procedure
        #====

        images = to_var(images)
        
        # Infer lower level z from data and concat into 32x8x8
        z_col_lst = []
        for i in xrange(0, 2):
            z_row_lst = []
            for j in xrange (0, 2):
                xs = images[:, :, i*seg_length:(i+1)*seg_length, j*seg_length:(j+1)*seg_length]
                z_volume = inf_bot(xs, take_pre=True)
                z_row_lst.append(z_volume)
            z_col = torch.cat(z_row_lst, 3)
            z_col_lst.append(z_col)
        z_bot = torch.cat(z_col_lst, 2)
        
        # Feed discriminator z inferred from data ("real")
        d_out_top_real = d_top(z_bot)
        d_loss_top_real = gan_loss(pre_sig=d_out_top_real, real=True, D=True, use_penalty=True, grad_inp=z_bot, gamma=1.0)
        print 'disc score real:', d_out_top_real
        # print "d loss top real", d_loss_top_real
        
        # Update discrimnator with "real"
        d_top.zero_grad()
        d_loss_top_real.backward(retain_graph=True)
        d_top_optimizer.step()

        # Reconstruct x through lower level z (sanity check)
        rec_images_bot = torch.zeros(batch_size, NUM_CHANNELS, IMAGE_LENGTH, IMAGE_LENGTH)
        for seg in range(0, ns):
            i = seg / ns_per_dim
            j = seg % ns_per_dim
            z_volume = z_bot[:, :, i*4:(i+1)*4, j*4:(j+1)*4]
            x_seg_low_rec = gen_bot(z_volume, give_pre=True)
            rec_images_bot[:, :, i*seg_length:(i+1)*seg_length, j*seg_length:(j+1)*seg_length] = x_seg_low_rec.data
            
        # Reconstruction x through z without shortcut
        rec_full_bot = torch.zeros(batch_size, NUM_CHANNELS, IMAGE_LENGTH, IMAGE_LENGTH)
        for seg in range(0, ns):
            i = seg / ns_per_dim
            j = seg % ns_per_dim
            xs = images[:, :, i*seg_length:(i+1)*seg_length, j*seg_length:(j+1)*seg_length]
            z_seg_full = inf_bot(xs)
            x_seg_full_rec = gen_bot(z_seg_full)
            rec_full_bot[:, :, i*seg_length:(i+1)*seg_length, j*seg_length:(j+1)*seg_length] = x_seg_full_rec.data
        
        
        # #### Higher Level Inference
        # z_top = inf_top(z_bot)
        # 
        # # Reconstruct lower level z through higher level z
        # z_bot_rec = gen_top(z_top)
        # rec_loss_top = ((z_bot - z_bot_rec)**2).mean()
        # print 'rec_loss_top:', rec_loss_top
        # 
        # # Update higher inference and generator networks for reconstruction
        # inf_top.zero_grad()
        # gen_top.zero_grad()
        # d_top.zero_grad()
        # rec_loss_top.backward()
        # inf_top_optimizer.step()
        # gen_top_optimizer.step()

        #============GENERATION PROCESS========================$

        # Sample higher and lower z
        z_top = to_var(torch.randn(batch_size, nz_high))
        z_bot = gen_top(z_top)
        
        # Feed discrimnator fake z
        d_out_top_fake = d_top(z_bot)
        d_loss_top_fake = gan_loss(pre_sig=d_out_top_fake, real=False, D=True, use_penalty=True, grad_inp=z_bot, gamma=1.0)
        print 'disc score fake:', d_out_top_fake
        
        # Update discrimnator according to fake z
        d_top.zero_grad()
        d_top.zero_grad()
        d_loss_top_fake.backward(retain_graph=True)
        d_top_optimizer.step()
        
        # Compute generator loss and update
        g_loss_top = gan_loss(pre_sig=d_out_top_fake, real=False, D=False, use_penalty=False, grad_inp=None, gamma=None, bgan=True)
        gen_top.zero_grad()
        d_top.zero_grad()
        g_loss_top.backward()
        gen_top_optimizer.step()
        
        z_bot = to_var(z_bot.data)

        # Generate fake images using lower level
        fake_images = torch.zeros(batch_size, NUM_CHANNELS, IMAGE_LENGTH, IMAGE_LENGTH)
        for seg in range(0, ns):
            i = seg / ns_per_dim
            j = seg % ns_per_dim
            z_bot = z_bot.view(batch_size, 32, 8, 8)
            z_volume = z_bot[:, :, i*4:(i+1)*4, j*4:(j+1)*4]
            x_seg_high_rec = gen_bot(z_volume, give_pre=True)
            fake_images[:, :, i*seg_length:(i+1)*seg_length, j*seg_length:(j+1)*seg_length] = x_seg_high_rec.data
        

        # reconstruction_loss = ((rec_images_top - images.cpu().data)**2).mean()
        # print 'high level reconstruction_loss:', reconstruction_loss
                
        elapsed = timer() - start_time
        if elapsed > checkpoint_i * CHECKPOINT_INTERVAL:
            print 'Writing images and checkpoints'
            make_dir_if_not_exists(OUT_DIR)
            make_dir_if_not_exists(MODELS_DIR)
        
            save_image(denorm(rec_images_bot), os.path.join(OUT_DIR, 'rec_bot_images%04d.png' % checkpoint_i))
            save_image(denorm(fake_images), os.path.join(OUT_DIR, 'fake_images%04d.png' % checkpoint_i))
            save_image(denorm(images.data), os.path.join(OUT_DIR, 'real_images%04d.png' % checkpoint_i))
            save_image(denorm(rec_full_bot), os.path.join(OUT_DIR, 'rec_full_bot_images%04d.png' % checkpoint_i))
            # save_image(denorm(rec_images_top), os.path.join(OUT_DIR, 'rec_images_top%04d.png' % checkpoint_i))

        
            # Checkpoint
            print 'Saving generator...'
            torch.save(gen_top, os.path.join(MODELS_DIR, '%s_gentop.pt' % slurm_name))
            print 'Saving inference...'
            torch.save(inf_top, os.path.join(MODELS_DIR, '%s_inftop.pt' % slurm_name))
            print 'Saving discriminator...'
            torch.save(d_top, os.path.join(MODELS_DIR, '%s_dtop.pt' % slurm_name))

            checkpoint_i += 1
                

end_time = timer()
elapsed = end_time - start_time
print 'total time used (in seconds):', elapsed
print 'total time used (in minutes):', elapsed / 60.0
print 'total time used (in hours):', elapsed / 60.0 / 60.0
