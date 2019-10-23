from keras.datasets import mnist
from keras.utils import to_categorical
from keras.datasets import fashion_mnist
import numpy as np


def MNISTdata(test=False):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape(60000, 784)
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape(10000, 784)
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    if test is False:
        return train_images, train_labels, test_images, test_labels
    if test is True:
        return test_images, test_labels

def MNISTdata_subset(Networktype, subset):
    (_, _), (test_images, test_labels) = mnist.load_data()

    idx = np.array([], dtype='int64')
    for i in subset:
        idx = np.union1d(idx, np.where(test_labels==i))

    test_labels = test_labels[idx]
    test_images = test_images[idx]
    num_idx = int(idx.shape[0])

    test_images = test_images.reshape(num_idx, 784)
    test_images = test_images.astype('float32') / 255
    test_labels = to_categorical(test_labels, num_classes=10)

    if Networktype == 'OriginalNet':
        return test_images, test_labels
    if Networktype == 'SubNet':
        test_labels = test_labels[:, subset]
        return test_images, test_labels

def FashionMNISTdata(test = False):
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images.reshape(60000, 28, 28, 1)
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    if test is False:
        return train_images, train_labels, test_images, test_labels
    if test is True:
        return test_images, test_labels

def FashionMNISTdata_subset(Networktype, subset):
    (_, _), (test_images, test_labels) = fashion_mnist.load_data()

    idx = np.array([], dtype='int64')
    for i in subset:
        idx = np.union1d(idx, np.where(test_labels == i))

    test_labels = test_labels[idx]
    test_images = test_images[idx]
    num_idx = int(idx.shape[0])

    test_images = test_images.reshape(num_idx, 28, 28, 1)
    test_images = test_images.astype('float32') / 255
    test_labels = to_categorical(test_labels, num_classes=10)

    if Networktype == 'OriginalNet':
        return test_images, test_labels
    if Networktype == 'SubNet':
        test_labels = test_labels[:, subset]
        return test_images, test_labels