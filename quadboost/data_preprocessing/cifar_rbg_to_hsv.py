from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np

import sys, os
sys.path.append(os.getcwd())
from quadboost.datasets import MNISTDataset, CIFAR10Dataset
from quadboost.utils import make_fig_axes


cifar = CIFAR10Dataset.load()
(Xtr, Ytr), (Xts, Yts) = cifar.get_train_test(shuffle=False)

def convert_to_hsv(X):
    X_r, X_g, X_b = (X[:,i,:,:,np.newaxis] for i in range(3))
    X_hsv = np.zeros_like(X, dtype=np.float32)
    X_channels_at_the_end = np.concatenate((X_r, X_g, X_b), axis=3)/255
    for i, x in enumerate(X_channels_at_the_end):
        pic = rgb2hsv(x)
        h, s, v = (pic[np.newaxis,:,:,i] for i in range(3))
        pic = np.concatenate((h,s,v), axis=0)
        X_hsv[i] = pic
    return X_hsv

Xtr_hsv = convert_to_hsv(Xtr)
Xts_hsv = convert_to_hsv(Xts)

cifar_hsv = CIFAR10Dataset(Xtr_hsv, Ytr, Xts_hsv, Yts)
cifar_hsv.save('cifar10hsv.pkl')


# cifar_hsv = CIFAR10Dataset.load('cifar10hsv.pkl')
# (Xtr, Ytr), (Xts, Yts) = cifar_hsv.get_train_test(shuffle=False)

cifar_grey = CIFAR10Dataset(Xtr[:,2], Ytr, Xts[:,2], Yts, shape=(32,32))
print(cifar_grey.Xtr.shape)
cifar_grey.save('cifar10grey.pkl')

cifar_sv = CIFAR10Dataset(Xtr[:,1:], Ytr, Xts[:,1:], Yts, shape=(2,32,32))
print(cifar_sv.Xtr.shape)
cifar_sv.save('cifar10sv.pkl')

# Xtr_hue = Xtr[:,0]
# print(np.max(Xtr_hue))
# print(np.min(Xtr_hue))
# Xtr_cs = np.concatenate((Xtr[:,1:], np.cos(Xtr_hue*2*np.pi), np.sin(Xtr_hue*2*np.pi)), axis=1)
# Xts_hue = Xts[:,0]


### To plot CIFAR10

# def plot_cifar(Xtr):
#     images = []
#     N = 20
#     for i, im in zip(range(N), Xtr):
#         im = im.reshape(3,32,32,1)
#         im = np.concatenate((im[0],im[1],im[2]), axis=2)
#         images.append(im)

#     fig, axes = make_fig_axes(len(images))
#     for im, ax in zip(images, axes):
#         ax.imshow(im)
#     plt.show()

# # plot_cifar(X)
# fig, axes = make_fig_axes(20)
# for im, ax in zip(X, axes):
#     ax.imshow(hsv_to_rgb(im))
# plt.show()
