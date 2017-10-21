import sys
sys.path.insert(0, '/u/lambalex/.local/lib/python2.7/site-packages/torch-0.2.0+4af66c4-py2.7-linux-x86_64.egg')
import torch
import numpy as np
from torch.autograd import Variable, grad
from utils import to_var, denorm
from torchvision.utils import save_image

'''
Trains a higher level model in isolation using a pre-trained generator and inference network from the lower level.  


'''

GB = torch.load('saved_models/63205_Gbot.pt')
IB = torch.load('saved_models/63205_Ibot.pt')

IB = IB.cuda()
GB = GB.cuda()

pacman_data = np.load('pacman_data_20k.npy')


for t in range(0,5):
    pacman_frames = to_var(torch.from_numpy(pacman_data[t,0:0+128,:,:,:]))

    #enc = to_var(torch.randn(128,512,4,4)) * 0.2
    enc = IB(pacman_frames,take_pre=True)

    print enc.size()

    dec = GB(enc, give_pre=True)

    print dec.size()

    new_enc = enc
    

    for k in range(0,1):

        new_dec = GB(new_enc, give_pre=True)

        loss = ((new_dec - pacman_frames)**2).mean()

        print k
        print loss

        new_enc = new_enc - 100.0 * grad(loss, new_enc)[0]

        print "z norm", new_enc.norm(2)

        new_enc = to_var(new_enc.data)

    save_image(denorm(new_dec.data), 'derp_%d.png' % t)

#rdir = 0.01 * to_var(torch.randn(128,64))

#pacman_frames = to_var(torch.from_numpy(pacman_data[t,0:0+128,:,:,:]))

#new_dec = pacman_frames

#for i in range(0,10):
#    enc = IB(new_dec)

#    new_enc = enc + rdir

#    new_dec = GB(new_enc)

#    save_image(denorm(new_dec.data), 'derp__%d.png' % i)



