import torch
from utils import to_var

batch_size = 100

real_labels = to_var(torch.ones(batch_size))
fake_labels = to_var(torch.zeros(batch_size))
boundary_labels = to_var(0.5 * torch.ones(batch_size))


def ls_loss(lst, target):

    loss = 0.0

    for v in lst: 
        if target == 1:
            t = to_var(torch.ones(v.size()))
        elif target == 0:
            t = to_var(torch.zeros(v.size()))
        elif target == 0.5:
            t = to_var(0.5 * torch.ones(v.size()))
        else:
            raise Exception()

        l = ((v - t)**2)

        if len(l.size()) == 2:
            l = l.mean(1)
        elif len(l.size()) == 4:
            l = l.mean(1).mean(1).mean(1)

        print l.size()

        loss += l

    return loss

#def orig_loss(lst, target):

#    loss = 0.0

#    for v in lst:
#        if target == 1:
#            loss += 
#        elif target == 0:
#            loss += 
#        else:
#            raise Exception()

