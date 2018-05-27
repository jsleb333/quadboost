import os, struct
import numpy as np
import matplotlib.pyplot as plt
from datasets import path_to
import json
# File paths, names and format
filename_images_train = 'train-images-idx3-ubyte'
filename_labels_train = 'train-labels-idx1-ubyte'
filename_images_test = 't10k-images-idx3-ubyte'
filename_labels_test = 't10k-labels-idx1-ubyte'
data_type_header, data_bytes_header = 'I', 4
data_type_labels, data_bytes_labels = 'B', 1
data_type_images, data_bytes_images = 'B', 1


def load_data(filename, N):
    with open(filename, 'rb') as file:
        header_size = 4 * data_bytes_header
        header_format = '>' + data_bytes_header*data_type_header #'>IIII' is the format of the data. '>' means big-endian, and 'I' means unsigned integer
        header = struct.unpack(header_format, file.read(header_size))

        magic_number, nber_images, nber_rows, nber_columns = header
        nber_pixels = nber_columns * nber_rows
        images = struct.unpack(
                        '>' + nber_pixels*data_type_images*N,
                        file.read(data_bytes_images*nber_pixels*N))
        images = np.array(images).reshape((N, 28, 28))
        return header, images


def load_labels(filename, N):
	with open(filename, 'rb') as labels:
		header_size = 2 * data_bytes_header
		header_format = '>' + 2 * data_type_header
		header = struct.unpack(header_format, labels.read(header_size))
		labels = struct.unpack('>'+data_bytes_labels*data_type_labels*N,labels.read(data_bytes_labels*N))
		return header, labels


def visualize_mnist(X, Y):
    for x, y in zip(X, Y):
        plt.imshow(x.reshape((28,28)), cmap='gray_r',)
        plt.title(y)
        plt.show()


def to_one_hot(Y):
    labels = set(Y)
    n_classes = len(labels)
    Y_one_hot = np.zeros((len(Y), n_classes))
    for i, label in enumerate(Y):
        Y_one_hot[i,label] = 1
    
    return Y_one_hot


def load_mnist(Ntr=60000, Nts=10000):
    mnist_path = path_to('mnist') + '\\raw\\'
    h, Xtr = load_data(mnist_path + filename_images_train, Ntr)
    h, Xts = load_data(mnist_path + filename_images_test, Nts)
    h, Ytr = load_labels(mnist_path + filename_labels_train, Ntr)
    h, Yts = load_labels(mnist_path + filename_labels_test, Nts)

    return (Xtr, Ytr), (Xts, Yts)


def load_encodings(encoding_name, convert_to_int=False):
    with open('./encodings.json') as file:
        encodings = json.load(file)[encoding_name]
    if convert_to_int:
        encodings = {int(label):encoding for label, encoding in encodings.items()}

    return encodings


def load_verbal_encodings(encoding_name):
    with open('./verbal_encodings.json') as file:
        verbal_encodings = json.load(file)[encoding_name]
    
    labels = sorted(set(l for f, [ones, zeros] in verbal_encodings.items() for l in ones+zeros))
    encodings = {l:-np.ones(len(verbal_encodings)) for l in labels}
    for i, (feature, [ones, zeros]) in enumerate(verbal_encodings.items()):
        for label in ones:
            encodings[label][i] = 1
        for label in zeros:
            encodings[label][i] = 0
    
    return encodings


if __name__ == '__main__':
    # (Xtr, Ytr), _ = load_mnist(Ntr=1000, Nts=0)

    # visualize_mnist(Xtr[-5:], Ytr[-5:])
    encodings = load_verbal_encodings('mario')
    print(encodings)