try:
    from quadboost.datasets.datasets import ImageDataset, MNISTDataset
    import datasets
except ModuleNotFoundError:
    from .datasets import ImageDataset, MNISTDataset
