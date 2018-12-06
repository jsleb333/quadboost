import numpy as np
import torch
from torchvision.transforms import RandomAffine
import torchvision.transforms.functional as tf
import torch
# def my_call(self, img):
#     """
#         img (PIL Image): Image to be transformed.

#     Returns:
#         Tuples of params.
#         PIL Image: Affine transformed image.
#     """
#     ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
#     return ret, tf.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor)
# RandomAffine.__call__ = my_call


from PIL.Image import BICUBIC

from quadboost.datasets import MNISTDataset
from quadboost.weak_learner.random_convolution import plot_images
from quadboost.utils import timed


def extend_mnist(Xtr, Ytr, N=1000, degrees=15, scale=(.85,1.11), shear=15):
    print(Xtr.dtype)
    Xtr_torch = torch.from_numpy(Xtr).reshape((-1,1,28,28))
    print(Xtr_torch.dtype)
    AffineTransform = RandomAffine(degrees=degrees, scale=scale, shear=shear)

    ex_Xtr = np.zeros((N, 28, 28), dtype=np.int32)
    ex_Ytr = np.zeros((N,), dtype=np.int32)
    for i in range(N):
        idx = np.random.randint(Xtr.shape[0])
        X = Xtr_torch[idx]
        X_pil = tf.pad(tf.to_pil_image(X),3)
        # params, X_transform = AffineTransform(X_pil)
        X_transform = AffineTransform(X_pil)
        X_transform = tf.to_tensor(tf.crop(X_transform, 3, 3, 28, 28)).numpy().reshape(28,28)
        # trans_title = f'trans-d={params[0]:.2f}-scale={params[2]:.2f}-shear={params[3]:.2f}'
        # plot_images([Xtr[_].reshape(28,28), X_transform], ['orig', trans_title])
        ex_Xtr[i] = X_transform
        ex_Ytr[i] = Ytr[idx]

    return np.concatenate((Xtr, ex_Xtr)), np.concatenate((Ytr, ex_Ytr))


if __name__ == '__main__':
    mnist = MNISTDataset.load()
    Xtr, Ytr = mnist.get_train(center=False, reduce=False)
    Xtr, Ytr = timed(extend_mnist)(Xtr, Ytr, N=10)
    print(Xtr.dtype)
    # plot_images([X.reshape(28,28) for X in Xtr[:10]], [y for y in Ytr[:10]])
