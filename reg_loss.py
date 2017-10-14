'''
Loss corresponding to "stabilizing gans by regularization"

'''
import sys
sys.path.insert(0, '/u/lambalex/.local/lib/python2.7/site-packages/torch-0.2.0+4af66c4-py2.7-linux-x86_64.egg')
import torch
from torch.autograd import grad, Variable
import numpy as np

def gan_loss(pre_sig, real, D, use_penalty,grad_inp=None):

    p = torch.sigmoid(pre_sig)

    if use_penalty:
        gv = grad(outputs=pre_sig.sum(),inputs=grad_inp,create_graph=True,retain_graph=True)[0]**2
        #print "gv", gv
        if len(gv.size()) == 4:
            gv = gv.sum(dim=(1,2,3))
        elif len(gv.size()) == 2:
            gv = gv.sum(dim=1)
        else:
            raise Exception('invalid shape')

    if real == True and D == True:
        cl = -torch.log(p).mean()
 
        if use_penalty:
            penalty = ((1 - p)**2 * gv).mean()
        else:
            penalty = 0.0

        print "penalty", penalty
        print "cl", cl

        loss = cl + penalty
    elif real == False and D == True:
        cl = -torch.log(1-p).mean()

        if use_penalty:
            penalty = (p**2 * gv).mean()
        else:
            penalty = 0.0

        print "penalty", penalty
        print "cl", cl

        loss = cl + penalty
    elif real == True and D == False:
        assert grad_inp is None

        loss = -torch.log(1-p).mean()

    elif real == False and D == False:
        assert grad_inp is None
        
        loss = -torch.log(p).mean()

    return loss

if __name__ == "__main__":


    for pens in [True,False]:
        print "penalty", pens
        x = Variable(torch.from_numpy(np.array([[2.5,0.0,0.0,0.1]])), requires_grad=True)
        pre_sig = (x**3).sum(dim=1)
        print "p", torch.sigmoid(pre_sig)
        l = gan_loss(pre_sig, False, True, use_penalty=pens,grad_inp=x)
        g = grad(l, x)[0].norm(2)
        print "loss", l
        print "grad", g




