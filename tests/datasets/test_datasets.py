import unittest as ut
import numpy as np
import os, sys
sys.path.append(os.getcwd())

from quadboost.datasets import ImageDataset


class TestImageDataset(ut.TestCase):
    def setUp(self):
        pass

    def test_shape_correctly_infered(self):
        Xtr = np.arange(7840).reshape(10, 28, 28)
        Ytr = np.ones(10)
        some_dataset = ImageDataset(Xtr, Ytr, None, None)

        self.assertEqual(some_dataset.shape, (28,28))

    def test_images_correctly_flattened(self):
        Xtr = np.arange(7840).reshape(10, 28, 28)
        Ytr = np.ones(10)
        Xts = np.arange(78400).reshape(100, 28, 28)
        Yts = np.ones(100)
        some_dataset = ImageDataset(Xtr, Ytr, Xts, Yts)
        self.assertEqual(some_dataset.Xtr.shape, (10, 784))
        self.assertEqual(some_dataset.Xts.shape, (100, 784))

    def test_get_train_valid_restitutes_shape(self):
        Xtr = np.arange(7840).reshape(10, 28, 28)
        Ytr = np.ones(10)
        some_dataset = ImageDataset(Xtr, Ytr, None, None)

        (Xtr, Ytr), (Xval, Yval) = some_dataset.get_train_valid(3, False, False, False)
        self.assertEqual(Xtr.shape[1:], (28, 28))
        self.assertEqual(Xval.shape[1:], (28, 28))

    def test_get_train_valid_valid_as_int(self):
        Xtr = np.arange(7840).reshape(10, 28, 28)
        Ytr = np.ones(10)
        some_dataset = ImageDataset(Xtr, Ytr, None, None)

        valid = 3
        (Xtr, Ytr), (Xval, Yval) = some_dataset.get_train_valid(valid, False, False, False)
        self.assertEqual(Xtr.shape[0], 7)
        self.assertEqual(Ytr.shape[0], 7)
        self.assertEqual(Xval.shape[0], 3)
        self.assertEqual(Yval.shape[0], 3)

    def test_get_train_valid_valid_as_float(self):
        Xtr = np.arange(7840).reshape(10, 28, 28)
        Ytr = np.ones(10)
        some_dataset = ImageDataset(Xtr, Ytr, None, None)

        valid = .35
        (Xtr, Ytr), (Xval, Yval) = some_dataset.get_train_valid(valid, False, False, False)
        self.assertEqual(Xtr.shape[0], 7)
        self.assertEqual(Ytr.shape[0], 7)
        self.assertEqual(Xval.shape[0], 3)
        self.assertEqual(Yval.shape[0], 3)

    def test_get_train_valid_centered(self):
        Xtr_orig = np.arange(2).reshape(2, 1, 1)
        Ytr_orig = np.ones(10)
        some_dataset = ImageDataset(Xtr_orig, Ytr_orig, None, None)

        valid = 0
        (Xtr, Ytr), (Xval, Yval) = some_dataset.get_train_valid(valid, center=True, reduce=False, shuffle=False)
        self.assertEqual(Xtr[0,0,0], -.5)
        self.assertEqual(Xtr[1,0,0], .5)

    def test_get_train_valid_reduced(self):
        pass

    def test_get_train_valid_shuffled(self):
        pass

    def tearDown(self):
        pass


if __name__ == "__main__":
    ut.main()
