import torch
from utils import to_var
#from gradient_penalty import gradient_penalty

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

        l = ((v - t)**2).mean()

        loss += l

    return loss

'''
def reg_loss(lst, gd, rf, inp = None):

    assert len(lst) == 0

    p = lst[0]

    loss = 0.0

    if gd == "d":
        assert inp is not None
        gp = gradient_penalty(p.sum(), inp)
        if rf == "r":
            pass
        elif rf == "f":
            pass
        else:
            raise Exception()
    elif gd == "g":
            pass
        if rf == "r":
            pass
        elif rf == "f":
            pass
        else:
            raise Exception()
    else:
        raise Exception()
'''







