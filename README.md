# QuadBoost.MH

This project aims to implement a performant boosting algorithm to classify images based on the quadratic loss.
The project is written in `Python` and tries to follow the philosophy of the `scikit-learn` project.

The current development of the project focuses on the MNIST Dataset.

## Getting started

To make the program work, you have to download the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).
The module `mnist_dataset.py` provides resources to handle the dataset.
The function `download_mnist` will download MNIST to your computer in the specified directory.
Alternatively, if you already have MNIST downloaded and unzipped elsewhere on your computer, you can skip this step and only provide the path to the dataset into the function `load_raw_mnist`.
Once this is done, the `main.py` script present a typical training ready to launch with many options.

### MNIST Dataset

The `mnist_dataset.py` file provides the necessary resources to unpack the raw dataset.
It also provides a class `MNISTDataset` which handle the dataset and can center/reduce it if desired.
This class can pickle the dataset, which make it faster to load in subsequent uses.
This step is required to run the minimal working examples.

### Prerequisites

This project relies on the following `Python` libraries:
- scikit-learn
- numpy
- matplotlib
- scikit-image (optional, used in `mnist_ideals/ideal_preprocessing.py`)
- tblib (optional, used in `utils.py`)

## Implementation description

The file `quadboost.py` provides an implementation of a general QuadBoost algorithm, with other specific implementations (QuadBoost.MH and QuadBoost.MHCR).
A `main()` function with minimal working example is also provided.

The module `weak_learner` provides some weak learners to be used with QuadBoost, such as a `MulticlassDecisionStump`.
All weak learners can be used as standalone.
A `Cloner` parent class is provided to facilitate the implementations of other weak learners that can easily be passed to the QuadBoost algorithm.

The file `label_encoder.py` provides an implementation of LabelEncoder and inherited classes.
These `LabelEncoder` can transform a set of labels into vectors encoding the classes, such as _one-hot_ encoding or _all-pairs_ encodings.
The class provides a method to encode and decode labels, and supports custom encodings.
Many examples of such custom encodings are presented in the `encodings.json` file, such as idealized MNIST characters, or mean haar transformed pictures.

The module `data_preprocessing` provides scripts to preprocess MNIST to extract features.
Current version only implements 2D Haar wavelet transform on features.

The boosting algorithm works with the help of callbacks on each step of the iteration.
Callbacks are handled by a `IteratorManager` which appropriately calls functions on beginnig of iteration, beginning of step, end of step, end of iteration and on exception exit.
Callbacks include `BreakCallbacks` which can end the iteration on various conditions, `ModelCheckpoint` and `CSVLogger` which saves the model or the logs and `Progession` which outputs formatted information on the training steps.
