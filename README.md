# QuadBoost.MH

This project aims to implement a performant boosting algorithm to classify images based on the quadratic loss. The project is written in `Python` and tries to follow the philosophy of the `scikit-learn` project.

The current development of the project focuses on the MNIST Dataset.

## Getting started

To make the program work, you have to download the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) to a folder, and write the path to the directory in the header of the file `mnist_dataset.py`.

Alternatively, if you already have `mnist` on your computer, you can create a file `datasets.py` which defines the function
```python
def path_to(dataset):
    if dataset == 'mnist':
        return '/path/to/mnist dataset/'
```

In the same folder, add an empty `__init__.py` file.
Then add the path to this file to your `PYTHONPATH`.

## MNIST Dataset

The `mnist_dataset.py` file provides the necessary resources to unpack the raw dataset. It also provides a class `MNISTDataset` which handle the dataset and can center/reduce it if desired. This class can pickle the dataset, which make it faster to load in subsequent uses. This step is required to make the minimal working examples to work.

## Prerequisites

This project relies on the following `Python` libraries:
- scikit-learn
- numpy
- matplotlib (optional, used for visualization)
- pytorch (optional, used for accelerating decision stumps)

## Implementation description

The file `quadboost.py` provides an implementation of a general QuadBoost algorithm, with other specific implementations (QuadBoost.MH and QuadBoost.MHCR).
A `main()` function with minimal working example is also provided.

The module `weak_learner` provides some weak learners to be used with QuadBoost. All weak learners can be used as standalone. A `cloner` wrapper is provided to facilitate the implementations of other weak learners that can easily be passed to the QuadBoost algorithm.

The file `label_encoder.py` provides an implementation of LabelEncoder and inherited classes. These LabelEncoder can transform a set of labels into vectors encoding the classes, such as _one-hot_ encoding or _all-pairs_ encodings. The class provides a method to encode and decode labels, and support custom encodings. Examples of such custom encodings are presented in the `encodings.json` file.



