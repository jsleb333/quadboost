import sys, os
sys.path.append(os.getcwd())
import struct
import numpy as np
import matplotlib.pyplot as plt
import gzip, urllib, io # To download MNIST

import pickle as pkl
from sklearn.preprocessing import StandardScaler
import warnings
from time import time

try:
    from datasets_path import path_to
except:
    path_to = {'mnist':"quadboost/data/mnist/"}
mnist_raw = os.path.join(path_to['mnist'], 'raw/')

# File paths, names and format
filename_images_train = 'train-images-idx3-ubyte'
filename_labels_train = 'train-labels-idx1-ubyte'
filename_images_test = 't10k-images-idx3-ubyte'
filename_labels_test = 't10k-labels-idx1-ubyte'
data_type_header, data_bytes_header = 'I', 4
data_type_labels, data_bytes_labels = 'B', 1
data_type_images, data_bytes_images = 'B', 1


def download_mnist(filepath=mnist_raw):
    os.makedirs(filepath, exist_ok=True)
    for filename in [filename_images_train,
                     filename_images_test,
                     filename_labels_train,
                     filename_labels_test]:
        url = f'http://yann.lecun.com/exdb/mnist/{filename}.gz'
        content = urllib.request.urlopen(url)
        with open(filepath + filename, 'wb') as file:
            file.write(gzip.decompress(content.read()))


def load_raw_data(filename, N):
    with open(filename, 'rb') as file:
        header_size = 4 * data_bytes_header
        header_format = '>' + data_bytes_header*data_type_header #'>IIII' is the format of the data. '>' means big-endian, and 'I' means unsigned integer
        header = struct.unpack(header_format, file.read(header_size))

        magic_number, nber_images, nber_rows, nber_columns = header
        nber_pixels = nber_columns * nber_rows
        images = struct.unpack(
                        '>' + nber_pixels*data_type_images*N,
                        file.read(data_bytes_images*nber_pixels*N))
        images = np.array(images).reshape((N, 784))
        return header, images


def load_raw_labels(filename, N):
	with open(filename, 'rb') as labels:
		header_size = 2 * data_bytes_header
		header_format = '>' + 2 * data_type_header
		header = struct.unpack(header_format, labels.read(header_size))
		labels = struct.unpack('>'+data_bytes_labels*data_type_labels*N,labels.read(data_bytes_labels*N))
		return header, np.array(labels)


def load_mnist(Ntr=60000, Nts=10000, path=mnist_raw):
    t = time()
    mnist_path = path
    if filename_images_train not in os.listdir(mnist_path):
        print('Downloading mnist...')
        download_mnist()
        print('Download finished.')
    h, Xtr = load_raw_data(mnist_path + filename_images_train, Ntr)
    h, Xts = load_raw_data(mnist_path + filename_images_test, Nts)
    h, Ytr = load_raw_labels(mnist_path + filename_labels_train, Ntr)
    h, Yts = load_raw_labels(mnist_path + filename_labels_test, Nts)
    print(f'Loaded MNIST in {time()-t:.2f}s.')

    return (Xtr, Ytr), (Xts, Yts)


if __name__ == "__main__":
    load_mnist()
