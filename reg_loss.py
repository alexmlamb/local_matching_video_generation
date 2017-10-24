
'''
Loss corresponding to "stabilizing gans by regularization"

'''
import sys
sys.path.insert(0, '/u/lambalex/.local/lib/python2.7/site-packages/torch-0.2.0+4af66c4-py2.7-linux-x86_64.egg')
import torch
from torch.autograd import grad, Variable
import numpy as np

def gan_loss(pre_sig, real, D, use_penalty,grad_inp=None,gamma=1.0,bgan=False):

    p = torch.sigmoid(pre_sig)

    if use_penalty:
        gv = grad(outputs=pre_sig.sum(),inputs=grad_inp,create_graph=True,retain_graph=True)[0]**2
        #print "gv", gv
        if len(gv.size()) == 4:
            gv = gv.sum(1).sum(1).sum(1)
        elif len(gv.size()) == 2:
            gv = gv.sum(dim=1)
        else:
            raise Exception('invalid shape')

        if len(p.size()) == 4:
            p = p.mean(1).mean(1).mean(1)

        #print "gv shape", gv.size()
        #print "p shape", p.size()

    if real == True and D == True:
        cl = -torch.log(p).mean()
 
        if use_penalty:
            penalty = (gv).mean()#((1 - p)**2 * gv).mean()
        else:
            penalty = 0.0

        #print "penalty", penalty
        #print "cl", cl

        loss = cl + penalty*gamma
    elif real == False and D == True:
        cl = -torch.log(1-p).mean()

        if use_penalty:
            penalty = (gv).mean()#(p**2 * gv).mean()
        else:
            penalty = 0.0

        #print "penalty", penalty
        #print "cl", cl

        loss = cl + penalty*gamma
    elif real == True and D == False:
        assert grad_inp is None
        assert use_penalty == False

        if bgan:
            loss = (torch.log(p/(1-p))**2).mean()
        else:
            loss = -torch.log(1-p).mean()

    elif real == False and D == False:
        assert grad_inp is None
        assert use_penalty == False
        
        if bgan:
            loss = (torch.log(p/(1-p))**2).mean()
        else:
            loss = -torch.log(p).mean()

    return loss


def gan_loss_multi(pre_sig_lst, real, D, use_penalty,grad_inp=None,gamma=1.0):


    loss = 0.0

    for pre_sig in pre_sig_lst:
        loss += gan_loss(pre_sig, real, D, use_penalty,grad_inp,gamma)

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





