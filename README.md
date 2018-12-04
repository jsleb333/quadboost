# QuadBoost.MH

This project aims to implement a performant boosting algorithm to classify images based on the quadratic loss.
Its inherent multiclass nature makes it a good alternative to classical boosting algorithms with simple class reductions.
The project is written in `Python` and tries to follow the philosophy of the `scikit-learn` project.

This package not only provides the QuadBoost.MH algorithm, but also a complete framework to boost any multiclass weak learners.
Therefore, it provides callbacks to customize the training, it comes with a variety of weak learners and uses outputs encoding.

The current development of the project focuses on the MNIST Dataset.

## Getting started

To make the minimal working examples of the program work, you have to download the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) or the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
The module `datasets` provides resources to easily handle datasets.
In the file `./quadboost/datasets/datasets.py`, you will find the functions `_generate_mnist_dataset` and `_generate_cifar10_dataset`.
These will automatically download the datasets, create a MNISTDataset or CIFAR10Dataset object and save it to the specified directory (which is `./quadboost/data/` by default).

Alternatively, if you already have MNIST or CIFAR-10 downloaded, you can create the file `./quadboost/datasets/datasets_path.py` containing a `dict` of the form:
```python
path_to = {'mnist':'path/to/mnist/raw/',
           'cifar10':'path/to/cifar10/raw'}
```
and the functions should create the datasets objects without downloading.

### Dataset

The `datasets.py` file provides the classes `MNISTDataset` and `CIFAR10Dataset`, which handle the datasets and can center and/or reduce them if desired.
This class can pickle the dataset, which make it faster to load in subsequent uses.
The use of the datasets is required to run the minimal working examples.

### Prerequisites

This project relies on the following `Python` libraries:
- scikit-learn
- numpy
- matplotlib
- pytorch (used in `weak_learner/random_convolution.py`)
- scikit-image (optional, used in `mnist_ideals/ideal_preprocessing.py`)
- tblib (optional, used in `utils/multiprocessing_utils.py`)
- colorama (optional, used in `utils/timed.py`)

## Implementation description

The file `quadboost.py` provides an implementation of a general QuadBoost algorithm, with other specific implementations (QuadBoost.MH and QuadBoost.MHCR).
A `main()` function with minimal working example is also provided.

The module `weak_learner` provides some weak learners to be used with QuadBoost, such as a `MulticlassDecisionStump` and a `MulticlassDecisionTree` based on the former.
Is also included a `RandomConvolution` feature extractor that wraps around a weak learner.
All weak learners can be used as standalone.
A `_WeakLearnerBase` parent class is provided to facilitate the implementations of other weak learners that can easily be passed to the QuadBoost algorithm.

The file `label_encoder.py` provides an implementation of LabelEncoder and inherited classes.
These `LabelEncoder` can transform a set of labels into vectors encoding the classes, such as _one-hot_ encoding or _all-pairs_ encodings.
The class provides a method to encode and decode labels, and supports custom encodings.
Many examples of such custom encodings are presented in the `encodings.json` file, such as idealized MNIST characters, or mean haar transformed pictures.

The module `data_preprocessing` provides scripts to preprocess MNIST to extract features.
Current version only implements 2D Haar wavelet transform on features.

The boosting algorithm works with the help of callbacks on each step of the iteration.
Callbacks are handled by a `CallbacksManagerIterator` which appropriately calls functions on beginnig of iteration, beginning of step, end of step, end of iteration and on exception exit.
Callbacks include `BreakCallbacks` which can end the iteration on various conditions, `ModelCheckpoint` and `CSVLogger` which saves the model or the logs and `Progession` which outputs formatted information on the training steps.
