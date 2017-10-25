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
LOWER_ONLY = True
REC_PENALTY = True

start_time = timer()


def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)


def torch_to_norm(zs):
    return zs.norm(2).data.cpu().numpy()[0]

if DATASET == 'mnist':
    batch_size = 100
    IMAGE_LENGTH = 28
    NUM_CHANNELS = 1
    print 'Loading MNIST dataset'
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])
    dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    nz = 64
    ns = 4
elif DATASET == 'lsun_bedroom':
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
else:
    raise ValueError('Unsupported dataset: %s' % DATASET)

NUM_DISC_OUTPUTS = 4
real_labels = to_var(torch.ones(batch_size, NUM_DISC_OUTPUTS))
fake_labels = to_var(torch.zeros(batch_size, NUM_DISC_OUTPUTS))
boundary_labels = to_var(0.5 * torch.ones(batch_size, NUM_DISC_OUTPUTS))


data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
ns_per_dim = int(sqrt(ns))

# for i, (images, _) in enumerate(data_loader):
#     print images.size()
# raise Exception('image size printed')    

seg_length = IMAGE_LENGTH / ns_per_dim

print "ns", ns

#(zL,xL) and (zR,xR)
from archs.mnist import Disc_Low
d_bot = Disc_Low(batch_size, seg_length, nz)

# from D_Top import D_Top
# d_top = D_Top(batch_size, nz*ns, nz, 256)
from archs.mnist import Disc_High
d_top = Disc_High(batch_size, nz*ns, nz, 256)

#(xL->zL) and (xR->zR)
from archs.mnist import Inf_Low
inf_bot = Inf_Low(batch_size, seg_length, nz)

#(zL->xL) and (zR->xR)
from archs.mnist import Gen_Low
gen_bot = Gen_Low(batch_size, seg_length, nz)

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

d_top_optimizer = torch.optim.Adam(d_top.parameters(), lr=0.0003)
d_bot_optimizer = torch.optim.Adam(d_bot.parameters(), lr=0.0003)
inf_bot_optimizer = torch.optim.Adam(inf_bot.parameters(), lr=0.0003)
gen_bot_optimizer = torch.optim.Adam(gen_bot.parameters(), lr=0.0003)
inf_top_optimizer = torch.optim.Adam(inf_top.parameters(), lr=0.0003)
gen_top_optimizer = torch.optim.Adam(gen_top.parameters(), lr=0.0003)

z_bot_norms = []
z_top_norms = []
checkpoint_i = 1
for epoch in range(200):
    for i, (images, _) in enumerate(data_loader):

        #batch_size = images.size(0)

        #====
        #Inference Procedure

        #gen_bot, then 

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
            zs = inf_bot(xs)
            
            # Feed discriminator real data
            # Discriminator on only x (not ALI)
            d_out_bot = d_bot(xs, zs)
            print 'd_out_bot.size():', d_out_bot.size()
            # d_loss_bot = ((d_out_bot - real_labels)**2).mean()
            d_loss_bot = gan_loss(pre_sig=d_out_bot, real=True, D=True, use_penalty=True, grad_inp=xs, gamma=1.0) + \
                gan_loss(pre_sig=d_out_bot, real=True, D=True, use_penalty=True, grad_inp=zs, gamma=1.0)

            # Generator loss pushing real data toward boundary
            # g_loss_bot = 1.0 * ((d_out_bot - boundary_labels)**2).mean()
            g_loss_bot = gan_loss(pre_sig=d_out_bot, real=True, D=False, use_penalty=False, grad_inp=None, gamma=None, bgan=True)

            # Add z norm penalty
            if Z_NORM_MULT is not None:
                g_loss_bot += Z_NORM_MULT * zs.norm(2)

            print "d loss bot inf", d_loss_bot

            # Reconstruct x through lower level z
            # Currently used for lower level generator learning
            reconstruction = gen_bot(inf_bot(xs, take_pre=True), give_pre=True)
            # reconstruction = gen_bot(zs)
            print 'zs.size():', zs.size()
            print 'xs.size():', xs.size()
            print 're.size():', reconstruction.size()
            rec_loss = ((reconstruction - xs)**2).mean()

            if REC_PENALTY:
                g_loss_bot += rec_loss
                print "training with reconstruction loss"
            else:
                print "training without low level reconstruction loss"

            print "reconstruction loss", rec_loss

            d_bot.zero_grad()
            d_loss_bot.backward(retain_graph=True)
            d_bot_optimizer.step()

            gen_bot.zero_grad()
            inf_bot.zero_grad()
            d_bot.zero_grad()
            g_loss_bot.backward(retain_graph=True)

            # Only do update 10% of the time
            # But still backprop every time?
            if random.uniform(0,1) < 0.1:
            # if random.uniform(0,1) < 1.0:
                gen_bot_optimizer.step()
                inf_bot_optimizer.step()

            z_bot_lst.append(zs)

        z_bot = torch.cat(z_bot_lst, 1)

        z_bot = Variable(z_bot.data)

        if not LOWER_ONLY:
            # Infer higher level z from data
            z_top = inf_top(z_bot)
            
            # Reconstruct lower level z through higher level z
            # Currently used for higher level generator learning
            z_bot_rec = gen_top(z_top)
            reconstruction_loss = ((z_bot - z_bot_rec)**2).mean()
            
            print "high level rec loss", reconstruction_loss
            
            # Discriminator on only lower z (not ALI)
            d_out_tops = d_top(z_bot)
            
            # Higher level discriminator now outputs a list, so sum over that list
            if SUM_DISC_OUTS:
                d_loss_top = 0
                g_loss_top = 0
                for d_out_top in d_out_tops:
                    # Using inferred lower level z's as real examples for discriminator
                    d_loss_top += 1.0 / len(d_out_tops) * ((d_out_top - real_labels)**2).mean()
                    g_loss_top += 1.0 / len(d_out_tops) * ((d_out_top - boundary_labels)**2).mean()
            else:
                d_loss_top = ((d_out_tops[-1] - real_labels)**2).mean()
                g_loss_top = ((d_out_tops[-1] - boundary_labels)**2).mean()
            
            print "d loss top inf", d_loss_top
            
            #d_loss_top += 0.1 * gradient_penalty(d_out_top.norm(2), z_bot)
            
            if REC_PENALTY:
                print "optimizing for high level rec loss"
                g_loss_top += reconstruction_loss
            else:
                print "not optimizing high level rec loss"
            
            d_top.zero_grad()
            d_loss_top.backward(retain_graph=True)
            d_top_optimizer.step()
            
            inf_top.zero_grad()
            gen_top.zero_grad()
            d_top.zero_grad()
            g_loss_top.backward(retain_graph=True)
            inf_top_optimizer.step()
            gen_top_optimizer.step()

        print "epoch", epoch
        #============GENERATION PROCESS========================$

        if not LOWER_ONLY:
            # Sample higher and lower z
            z_top = to_var(torch.randn(batch_size, nz))
            z_bot = gen_top(z_top)
            
            d_out_tops = d_top(z_bot)
            
            d_top.zero_grad()
            
            # Higher level discriminator now outputs a list, so sum over that list
            if SUM_DISC_OUTS:
                d_loss_top = 0
                for d_out_top in d_out_tops:
                    # Using sampled lower level z's as fake examples for discriminator
                    d_loss_top += 1.0 / len(d_out_tops) * ((d_out_top - fake_labels)**2).mean()
            else:
                d_loss_top += ((d_out_tops[-1] - fake_labels)**2).mean()
            
            d_loss_top.backward(retain_graph=True)
            d_top_optimizer.step()
            
            print "d loss top gen", d_loss_top
            
            # Consider down-weighting each element by the number in the list?
            if SUM_DISC_OUTS:
                g_loss_top = 0
                for d_out_top in d_out_tops:
                    # Generator loss pushing generated lower z's toward boundary
                    g_loss_top = 1.0 / len(d_out_tops) * ((d_out_top - boundary_labels)**2).mean()
            else:
                g_loss_top = ((d_out_tops[-1] - boundary_labels)**2).mean()
            
            gen_top.zero_grad()
            d_top.zero_grad()
            g_loss_top.backward(retain_graph=True)
            gen_top_optimizer.step()
            
            z_bot = Variable(z_bot.data)

        gen_x_lst = []
        for seg in range(0,ns):
            # Generate sampled x's
            if LOWER_ONLY:
                seg_z = to_var(torch.randn(batch_size, nz))
            else:
                seg_z = z_bot[:,seg*nz:(seg+1)*nz]
                
            seg_x = gen_bot(seg_z)

            gen_x_lst.append(seg_x)
            d_out_bot = d_bot(seg_x, seg_z)
            # Discriminator for generated x's (not ALI)
            # d_loss_bot = ((d_out_bot - fake_labels)**2).mean()
            d_loss_bot = gan_loss(pre_sig=d_out_bot, real=False, D=True, use_penalty=True, grad_inp=seg_x, gamma=1.0) + \
                gan_loss(pre_sig=d_out_bot, real=False, D=True, use_penalty=True, grad_inp=seg_z, gamma=1.0)

            print "d loss bot gen", d_loss_bot

            d_bot.zero_grad()
            d_loss_bot.backward(retain_graph=True)
            d_bot_optimizer.step()

            print "train with less g loss bot"

            # Generator loss pushing generated x's toward boundary
            # g_loss_bot = 1.0 * ((d_out_bot - boundary_labels)**2).mean()
            g_loss_bot = gan_loss(pre_sig=d_out_bot, real=True, D=False, use_penalty=False, grad_inp=None, gamma=None, bgan=True)
            
            gen_bot.zero_grad()
            d_bot.zero_grad()
            g_loss_bot.backward(retain_graph=True)

            # Only update generator 10% of time
            # But still backprop every time?
            if random.uniform(0,1) < 0.1:
                gen_bot_optimizer.step()

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
                zs = inf_bot(xs)
                xr = gen_bot(zs)
                rec_images_bot[:, :, i*seg_length:(i+1)*seg_length, j*seg_length:(j+1)*seg_length] = xr.data
                # x_bot_lst.append(xr)
                # z_bot_lst.append(zs)

            # rec_images_bot = torch.cat(x_bot_lst, 1)
            # 
            # rec_images_bot = rec_images_bot.view(rec_images_bot.size(0), NUM_CHANNELS, IMAGE_LENGTH, IMAGE_LENGTH)
            save_image(denorm(rec_images_bot), os.path.join(OUT_DIR, 'rec_images_bot%05d.png' % checkpoint_i))
        
            if not LOWER_ONLY:
                z_bot = torch.cat(z_bot_lst, 1)
                z_top = inf_top(z_bot)
                z_bot = gen_top(z_top)
            
                gen_x_lst = []
                for seg in range(0,ns):
                    seg_z = z_bot[:,seg*nz:(seg+1)*nz]
                    gen_x_lst.append(gen_bot(seg_z))
            
                rec_images_top = torch.cat(gen_x_lst, 1)
            
                rec_images_top = rec_images_top.view(rec_images_top.size(0), NUM_CHANNELS, IMAGE_LENGTH, IMAGE_LENGTH)
                save_image(denorm(rec_images_top.data), os.path.join(OUT_DIR, 'rec_images_top%05d.png' % checkpoint_i))
            
            # Checkpoint
            torch.save(gen_bot, os.path.join(MODELS_DIR, '%s_genbot.pt' % slurm_name))
            torch.save(d_bot, os.path.join(MODELS_DIR, '%s_dbot.pt' % slurm_name))
            torch.save(inf_bot, os.path.join(MODELS_DIR, '%s_infbot.pt' % slurm_name))
            
            if not LOWER_ONLY:
                torch.save(gen_top, os.path.join(MODELS_DIR, '%s_gentop.pt' % slurm_name))
                torch.save(d_top, os.path.join(MODELS_DIR, '%s_dtop.pt' % slurm_name))
                torch.save(inf_top, os.path.join(MODELS_DIR, '%s_inftop.pt' % slurm_name))
                
            checkpoint_i += 1
                

end_time = timer()
elapsed = end_time - start_time
print 'total time used (in seconds):', elapsed
print 'total time used (in minutes):', elapsed / 60.0
print 'total time used (in hours):', elapsed / 60.0 / 60.0
