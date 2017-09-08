
from scipy.misc import face
import numpy as np
import matplotlib.pyplot as plt
import gzip
import cPickle as pickle
import random

from skimage import data
from skimage.transform import swirl

#img = face()

#print img.shape
dataset = "mnist"
if dataset == "mnist":
    mn = gzip.open("/u/lambalex/data/mnist/mnist.pkl.gz")

    train, valid, test = pickle.load(mn)

    trainx,trainy = train


    #newtx = trainx[(trainy<2)]# | (trainy>8)]
    #newty = trainy[(trainy<2)]# | (trainy>8)]
    #trainx = newtx
    #trainy = newty

    validx,validy = valid
    testx, testy = test

    num_examples = trainx.shape[0]

    m = 784

print trainx[0].shape

img = trainx[random.randint(0,2000)].reshape(28,28)

plt.imshow(img, cmap=plt.cm.gray)
plt.show()

A = img.shape[0] / 10.0
w = 2.0 / img.shape[1]

shift = lambda x: A * np.sin(2.0*np.pi*x * w)

for i in range(img.shape[0]):
        img[:,i] = np.roll(img[:,i], int(shift(i)))

plt.imshow(img, cmap=plt.cm.gray)
plt.show()

image = trainx[random.randint(0,2000)].reshape(28,28)

swirled = swirl(image, rotation=0, strength=3, radius=25)

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3),
                               sharex=True, sharey=True,
                               subplot_kw={'adjustable':'box-forced'})

ax0.imshow(image, cmap=plt.cm.gray, interpolation='none')
ax0.axis('off')
ax1.imshow(swirled, cmap=plt.cm.gray, interpolation='none')
ax1.axis('off')

plt.show()


