import torch
from torch.autograd import Variable
import os, errno


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x,requires_grad=True)


def make_dir_if_not_exists(path):
    """Make directory if doesn't already exists"""
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
