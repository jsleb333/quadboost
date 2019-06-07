import pytest
import numpy as np
import os, sys
sys.path.append(os.getcwd())

from quadboost.datasets import ImageDataset


class TestImageDataset:
    def test_shape_correctly_infered(self):
        Xtr = np.arange(7840).reshape(10, 28, 28)
        Ytr = np.ones(10)
        some_dataset = ImageDataset(Xtr, Ytr, None, None)

        assert some_dataset.shape == (28, 28)

    def test_images_correctly_flattened(self):
        Xtr = np.arange(7840).reshape(10, 28, 28)
        Ytr = np.ones(10)
        Xts = np.arange(78400).reshape(100, 28, 28)
        Yts = np.ones(100)
        some_dataset = ImageDataset(Xtr, Ytr, Xts, Yts)
        assert some_dataset.Xtr.shape == (10, 784)
        assert some_dataset.Xts.shape == (100, 784)

    def test_get_train_valid_restitutes_shape(self):
        Xtr = np.arange(7840).reshape(10, 28, 28)
        Ytr = np.ones(10)
        some_dataset = ImageDataset(Xtr, Ytr, None, None)

        (Xtr, Ytr), (Xval, Yval) = some_dataset.get_train_valid(3, False, False, False)
        assert Xtr.shape[1:] == (28, 28)
        assert Xval.shape[1:] == (28, 28)

    def test_get_train_valid_valid_as_int(self):
        Xtr = np.arange(7840).reshape(10, 28, 28)
        Ytr = np.ones(10)
        some_dataset = ImageDataset(Xtr, Ytr, None, None)

        valid = 3
        (Xtr, Ytr), (Xval, Yval) = some_dataset.get_train_valid(valid, False, False, False)
        assert Xtr.shape[0] == 7
        assert Ytr.shape[0] == 7
        assert Xval.shape[0] == 3
        assert Yval.shape[0] == 3

    def test_get_train_valid_valid_as_float(self):
        Xtr = np.arange(7840).reshape(10, 28, 28)
        Ytr = np.ones(10)
        some_dataset = ImageDataset(Xtr, Ytr, None, None)

        valid = .35
        (Xtr, Ytr), (Xval, Yval) = some_dataset.get_train_valid(valid, False, False, False)
        assert Xtr.shape[0] == 7
        assert Ytr.shape[0] == 7
        assert Xval.shape[0] == 3
        assert Yval.shape[0] == 3

    def test_get_train_valid_centered(self):
        Xtr_orig = np.arange(2).reshape(2, 1, 1)
        # Xtr_orig = [[[0]], [[1]]] => mean = .5
        Ytr_orig = np.ones(2)
        some_dataset = ImageDataset(Xtr_orig, Ytr_orig, None, None)

        valid = 0
        (Xtr, Ytr), (Xval, Yval) = some_dataset.get_train_valid(valid, center=True, reduce=False, shuffle=False)

        assert Xtr[0,0,0] == -.5
        assert Xtr[1,0,0] == .5

    def test_get_train_valid_reduced(self):
        # Variance formula: E[(X - E[X])²]
        Xtr_orig = np.arange(2).reshape(2, 1, 1)
        # Xtr_orig = [[[0]], [[1]]] => variance = .25
        Ytr_orig = np.ones(2)
        some_dataset = ImageDataset(Xtr_orig, Ytr_orig, None, None)

        valid = 0
        (Xtr, Ytr), (Xval, Yval) = some_dataset.get_train_valid(valid, center=False, reduce=True, shuffle=False)

        # We find the multiplicative factor "a" by setting the variance to 1:
        #    E[(a*X - E[a*X])²] = 1
        #    [(0 - a/2)² + (1 - a/2)²] = 1
        #    a = 2
        assert Xtr[0,0,0] == 0
        assert Xtr[1,0,0] == 2

    def test_get_train_valid_centered_reduced(self):
        Xtr_orig = np.arange(2).reshape(2, 1, 1)
        # Xtr_orig = [[[0]], [[1]]] => mean = .5, variance = .25
        Ytr_orig = np.ones(2)
        some_dataset = ImageDataset(Xtr_orig, Ytr_orig, None, None)

        valid = 0
        (Xtr, Ytr), (Xval, Yval) = some_dataset.get_train_valid(valid, center=True, reduce=True, shuffle=False)

        assert Xtr[0,0,0] == -1
        assert Xtr[1,0,0] == 1

    def test_get_train_valid_shuffled(self):
        Xtr_orig = np.arange(10).reshape(10, 1, 1)*10
        Ytr_orig = -np.arange(10)
        some_dataset = ImageDataset(Xtr_orig, Ytr_orig, None, None)

        valid = 0
        (Xtr, Ytr), (Xval, Yval) = some_dataset.get_train_valid(valid, center=False, reduce=False, shuffle=True)

        assert not all(x_orig[0,0] == x[0,0] for x_orig, x in zip(Xtr_orig, Xtr))
        indices = (np.argwhere(Xtr == x[0,0])[0][0] for x in Xtr_orig)
        assert all(y == Ytr[idx] for idx, y in zip(indices, Ytr_orig))

    def test_get_train(self):
        Xtr_orig = np.arange(10).reshape(10, 1, 1)*10
        Ytr_orig = -np.arange(10)
        some_dataset = ImageDataset(Xtr_orig, Ytr_orig, None, None)

        Xtr, Ytr = some_dataset.get_train(center=False, reduce=False, shuffle=True)
        assert Xtr.size == 10
        assert Ytr.size == 10

    def test_get_test_no_args(self):
        Xtr_orig = np.arange(1,11).reshape(10, 1, 1) * 10
        Ytr_orig = -np.arange(10)
        Xts_orig = np.arange(1,11).reshape(10, 1, 1)*2
        Yts_orig = -np.arange(10) - 10
        some_dataset = ImageDataset(Xtr_orig, Ytr_orig, Xts_orig, Yts_orig)

        Xts, Yts = some_dataset.get_test()
        assert all(Xts_orig == Xts)

    def test_get_test_scale_with_different_datasets(self):
        Xtr_orig = np.arange(1,11).reshape(10, 1, 1) * 10
        Ytr_orig = -np.arange(10)
        Xts_orig = np.arange(1,11).reshape(10, 1, 1)*2
        Yts_orig = -np.arange(10) - 10
        some_dataset = ImageDataset(Xtr_orig, Ytr_orig, Xts_orig, Yts_orig)

        Xts_1, Yts_1 = some_dataset.get_test(center=True, reduce=True, scale_with=Xtr_orig)
        Xts_2, Yts_2 = some_dataset.get_test(center=True, reduce=True, scale_with=Xtr_orig[:5,:,:])
        assert all(Xts_1 != Xts_2)

    def test_transform_data_restitutes_shape(self):
        Xtr_orig = np.arange(10).reshape(10, 1, 1)*10
        Ytr_orig = -np.arange(10)
        some_dataset = ImageDataset(Xtr_orig, Ytr_orig, None, None)

        Xtr, Ytr = some_dataset.transform_data(some_dataset.Xtr, some_dataset.Ytr)
        assert Xtr.shape == Xtr_orig.shape

    def test_transform_data_handles_empty_datasets(self):
        Xtr_orig = np.arange(10).reshape(10, 1, 1)*10
        Ytr_orig = -np.arange(10)
        some_dataset = ImageDataset(Xtr_orig, Ytr_orig, None, None)

        Xtr, Ytr = some_dataset.transform_data(np.array([]), np.array([]))
        assert Xtr.shape == (0,1,1)
