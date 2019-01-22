import os
from urllib.request import urlretrieve
import tarfile as tar

import numpy as np
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from time import time

try:
    from datasets_path import path_to
except:
    path_to = {'cifar10':"quadboost/data/cifar10/"}
cifar10_raw = os.path.join(path_to['cifar10'], 'raw/')


def download_cifar10(filepath=cifar10_raw):
    """
    Args:
        filepath (str, optional): Directory containing CIFAR-10. Path is created if nonexistant. CIFAR-10 is downloaded if missing.
    Returns None.
    """
    url = 'https://www.cs.toronto.edu/~kriz/'
    tar_filename = 'cifar-10-binary.tar.gz'

    if filepath is None:
        # Set filepath to /home/USER/data/cifar10 or C:\Users\USER\data\cifar10
        filepath = os.path.join(os.path.expanduser('~'), 'data', 'cifar10')

    # Create path if it doesn't exist
    os.makedirs(filepath, exist_ok=True)

    # Download tarfile if missing
    if tar_filename not in os.listdir(filepath):
        urlretrieve(''.join((url, tar_filename)), os.path.join(filepath, tar_filename))
        print("Downloaded %s to %s" % (tar_filename, filepath))


def load_cifar10(filepath=cifar10_raw):
    """
    Args:
        filepath (str, optional): Directory containing CIFAR-10. Path is created if nonexistant. CIFAR-10 is downloaded if missing.

    Returns:
        train_images, train_labels, test_images, test_labels (numpy arrays of shape (n_examples, 3072) for images and of shape (n_examples) for labels). Pixel values are in the order red, blue, green.
    """
    tar_filename = 'cifar-10-binary.tar.gz'
    filenames = ['cifar-10-batches-bin/data_batch_1.bin',
                 'cifar-10-batches-bin/data_batch_2.bin',
                 'cifar-10-batches-bin/data_batch_3.bin',
                 'cifar-10-batches-bin/data_batch_4.bin',
                 'cifar-10-batches-bin/data_batch_5.bin',
                 'cifar-10-batches-bin/test_batch.bin']

    # Download tarfile if missing
    if tar_filename not in os.listdir(filepath):
        download_cifar10(filepath)

    t = time()
    # Load data from tarfile
    with tar.open(os.path.join(filepath, tar_filename)) as tar_object:

        # Each file contains 10 000 color images and 10 000 labels
        file_size = 10000 * (32*32*3) + 10000
        # There are 6 files (5 train and 1 test)
        buffer = np.zeros(file_size * 6, dtype='uint8')

        # The tar contains README's and other extraneous stuff
        files = [file for file in tar_object if file.name in filenames]

        # Load files in proper order
        files.sort(key=lambda member: member.name)

        for i, file in enumerate(files):
            bytes_object = tar_object.extractfile(file)
            buffer[i*file_size:(i+1)*file_size] = np.frombuffer(bytes_object.read(), 'B')

    # Examples are in chunks of 3,073 bytes: the first is the label, the 3072 others are the image
    labels = buffer[::3073]
    pixels = np.delete(buffer, np.arange(0, buffer.size, 3073))
    images = pixels.reshape(-1, 3072)

    # Split into train and test
    train_images, test_images = images[:50000], images[50000:]
    train_labels, test_labels = labels[:50000], labels[50000:]

    print(f'Loaded CIFAR-10 in {time()-t:.2f}s.')
    return (train_images, train_labels), (test_images, test_labels)


if __name__ == '__main__':
    (Xtr, Ytr), (Xts, Yts) = load_cifar10()
    print(Xtr.shape)
