import torch
import torch.autograd
import numpy as np
from torch.autograd import Variable, grad

def gradient_penalty(out, inp):
    
    gradients = grad(outputs = inp.norm(2), inputs = inp, create_graph=True, retain_graph=True, only_inputs=True)[0]

    pen = ((gradients.norm(2, dim=1) - 1)**2).mean() * 0.1

    return pen

if __name__ == "__main__":
    inp = Variable(torch.from_numpy(np.array([[2.0,2.0],[2.0,2.0]])))

    out = inp**2

    g = grad(gradient_penalty(out, inp), inp)

    print g


