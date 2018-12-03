try:
    from quadboost.datasets.datasets import ImageDataset, MNISTDataset
    import mnist_dataset
except ModuleNotFoundError:
    from .datasets import ImageDataset, MNISTDataset
