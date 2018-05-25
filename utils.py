from datasets import path_to
import os, struct
import numpy as np
import matplotlib.pyplot as plt

# File paths, names and format
filename_images_train = 'train-images-idx3-ubyte'
filename_labels_train = 'train-labels-idx1-ubyte'
filename_images_test = 't10k-images-idx3-ubyte'
filename_labels_test = 't10k-labels-idx1-ubyte'
data_type_header, data_bytes_header = 'I', 4
data_type_images, data_bytes_images = 'I', 4
data_type_images_unprocessed, data_bytes_images_unprocessed = 'B', 1
data_type_labels, data_bytes_labels = 'B', 1


def load_data(filename, N):
    with open(filename, 'rb') as file:
        header_size = 4 * data_bytes_header
        header_format = '>' + data_bytes_header*data_type_header #'>IIII' is the format of the data. '>' means big-endian, and 'I' means unsigned integer
        header = struct.unpack(header_format, file.read(header_size))

        magic_number, nber_images, nber_rows, nber_columns = header
        nber_pixels = nber_columns * nber_rows
        images = struct.unpack(
                        '>' + nber_pixels*data_type_images_unprocessed*N,
                        file.read(data_bytes_images_unprocessed*nber_pixels*N))
        images = np.array([[images[i+j*784] for i in range(784)] for j in range(N)])
        return header, images


def load_labels(filename, N):
	with open(filename, 'rb') as labels:
		header_size = 2 * data_bytes_header
		header_format = '>' + 2 * data_type_header
		header = struct.unpack(header_format, labels.read(header_size))
		labels = struct.unpack('>'+data_bytes_labels*data_type_labels*N,labels.read(data_bytes_labels*N))
		return header, labels


def load_mnist(Ntr=60000, Nts=10000):
    mnist_path = path_to('mnist')
    h, Xtr = load_data(mnist_path+'\\raw\\'+filename_images_train, N=Ntr)
    h, Xts = load_data(mnist_path+'\\raw\\'+filename_images_test, N=Nts)
    h, Ytr = load_labels(mnist_path+'\\raw\\'+filename_labels_train, N=Ntr)
    h, Yts = load_labels(mnist_path+'\\raw\\'+filename_labels_test, N=Nts)

    return (Xtr, Ytr), (Xts, Yts)


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


if __name__ == '__main__':
    (Xtr, Ytr), _ = load_mnist(Ntr=1000, Nts=0)

    visualize_mnist(Xtr[-5:], Ytr[-5:])
