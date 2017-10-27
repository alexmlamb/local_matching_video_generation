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
Get samples from saved models
'''

BASELINE_ID = '80034'
HIGHER_ID = '80045'
LOWER_ID = '74342'
JOINT_ID = '82068'

BASELINE_MODELS_FOLDER = '/data/lisatmp4/nealbray/loc/cifar/gan/%s/saved_models' % BASELINE_ID
HIGHER_MODELS_FOLDER = '/data/lisatmp4/nealbray/loc/cifar/%s/saved_models' % HIGHER_ID
FIXED_LOWER_MODELS_FOLDER = '/data/lisatmp4/nealbray/loc/cifar/%s/saved_models' % LOWER_ID
JOINT_MODELS_FOLDER = '/data/lisatmp4/nealbray/loc/cifar/%s/saved_models' % JOINT_ID
SAMPLES_FOLDER = '/data/lisatmp4/nealbray/loc/cifar'


def get_model_samples(model_name, epoch):
    if model_name == 'baseline':
        return get_baseline_samples(epoch)
    elif model_name == 'higher':
        return get_higher_samples(epoch)
    elif model_name == 'joint':
        return get_joint_samples(epoch)
    else:
        raise ValueError('Invalid model name: %s' % model_name)
    
    
def get_baseline_samples(epoch):
    # filename = os.path.join(SAMPLES_FOLDER, 'baseline_samples%03d.pkl' % epoch)
    # if os.path.isfile(filename):
    #     print 'Samples file already exists. Loading...'
    #     with open(filename, 'rb') as f:
    #         samples = pickle.load(f)
    #         return samples
    
    print 'Drawing samples...'    
    gen = torch.load(os.path.join(BASELINE_MODELS_FOLDER, '%s_gen%03d.pt' % (BASELINE_ID, epoch)))
    nz = 256
    samples = []
    mn = 255
    mx = -255
    for _ in xrange(500):
        z = to_var(torch.randn(100, nz))
        fake_images = gen(z)
        fake_images_np = var_to_np(fake_images)
        mn = min(mn, fake_images_np.min())
        mx = max(mx, fake_images_np.max())
        for i in xrange(100):
            samples.append(fake_images_np[i, :, :, :])
    print 'min:', mn
    print 'max:', mx
    
    # print 'Saving samples to file...'
    # with open(filename, 'wb') as f:
    #     pickle.dump(samples, f)
            
    return samples
               

def get_higher_samples(epoch):
    print 'Drawing samples...' 
    gen_bot = torch.load(os.path.join(FIXED_LOWER_MODELS_FOLDER, '%s_genbot.pt' % LOWER_ID))
    gen_top = torch.load(os.path.join(HIGHER_MODELS_FOLDER, '%s_gentop%03d.pt' % (HIGHER_ID, epoch)))
    nz_high = 512
    
    samples = []
    seg_length = 16
    ns_per_dim = 2
    mn = 255
    mx = -255
    for _ in xrange(500):
        z_top = to_var(torch.randn(64, nz_high))
        z_bot = gen_top(z_top)
        fake_images = torch.zeros(64, 3, 32, 32)
        for seg in range(0, 4):
            i = seg / ns_per_dim
            j = seg % ns_per_dim
            z_bot = z_bot.view(64, 32, 8, 8)
            z_volume = z_bot[:, :, i*4:(i+1)*4, j*4:(j+1)*4]
            x_seg = gen_bot(z_volume, give_pre=True)
            fake_images[:, :, i*seg_length:(i+1)*seg_length, j*seg_length:(j+1)*seg_length] = x_seg.data
        fake_images_np = fake_images.cpu().numpy()
        
        mn = min(mn, fake_images_np.min())
        mx = max(mx, fake_images_np.max())
        for i in xrange(64):
            samples.append(fake_images_np[i, :, :, :])
    print 'min:', mn
    print 'max:', mx
    
    return samples


def get_joint_samples(epoch):
    print 'Drawing samples...' 
    gen_bot = torch.load(os.path.join(JOINT_MODELS_FOLDER, '%s_genbot%03d.pt' % (JOINT_ID, epoch)))
    gen_top = torch.load(os.path.join(JOINT_MODELS_FOLDER, '%s_gentop%03d.pt' % (JOINT_ID, epoch)))
    nz_high = 512
    
    samples = []
    seg_length = 16
    ns_per_dim = 2
    mn = 255
    mx = -255
    for _ in xrange(500):
        z_top = to_var(torch.randn(64, nz_high))
        z_bot = gen_top(z_top)
        fake_images = torch.zeros(64, 3, 32, 32)
        for seg in range(0, 4):
            i = seg / ns_per_dim
            j = seg % ns_per_dim
            z_bot = z_bot.view(64, 32, 8, 8)
            z_volume = z_bot[:, :, i*4:(i+1)*4, j*4:(j+1)*4]
            x_seg = gen_bot(z_volume, give_pre=True)
            fake_images[:, :, i*seg_length:(i+1)*seg_length, j*seg_length:(j+1)*seg_length] = x_seg.data
        fake_images_np = fake_images.cpu().numpy()
        
        mn = min(mn, fake_images_np.min())
        mx = max(mx, fake_images_np.max())
        for i in xrange(64):
            samples.append(fake_images_np[i, :, :, :])
    print 'min:', mn
    print 'max:', mx
    
    return samples
    
    
def var_to_np(x):
    return x.data.cpu().numpy()
