import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable, grad

'''
Initially just implement LSGAN on MNIST.  

Then implement a critic.  
'''

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])

mnist = datasets.MNIST(root='./data/', train=True, download=True, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=100, shuffle=True)

#Discriminator
D = nn.Sequential(
    nn.Linear(784, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1))

#Critic
C = nn.Sequential(
    nn.Linear(784, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1))

# Generator 
G = nn.Sequential(
    nn.Linear(64, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 784),
    nn.Tanh())

if torch.cuda.is_available():
    D.cuda()
    C.cuda()
    G.cuda()

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)
c_optimizer = torch.optim.Adam(C.parameters(), lr=0.0003)

for epoch in range(200):
    for i, (images, _) in enumerate(data_loader):

        batch_size = images.size(0)

        images = to_var(images.view(batch_size, -1))

        real_labels = to_var(torch.ones(batch_size))
        fake_labels = to_var(torch.zeros(batch_size))
        boundary_labels = to_var(0.5 * torch.ones(batch_size))

        outputs = D(images)
        d_loss_real = ((outputs - real_labels)**2).mean()
        real_score = outputs

        z = to_var(torch.randn(batch_size, 64))
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = ((outputs - fake_labels)**2).mean()
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake

        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        #z = to_var(torch.randn(batch_size, 64))
        #fake_images = G(z)
        outputs_critic = C(to_var(fake_images.data))
        c_loss = ((outputs_critic - to_var(fake_score.data))**2).mean()

        C.zero_grad()
        c_loss.backward()
        c_optimizer.step()

        print outputs_critic.mean()




