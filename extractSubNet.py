from keras import layers
from keras import models
import numpy as np


def MNIST_SubNet(model, subset, meanNodes=False):
    num_subset = len(subset)
    dense_W1, dense_b1 = model.layers[0].get_weights()
    dense_W2, dense_b2 = model.layers[1].get_weights()

    init_W1 = np.where(abs(dense_W1) > 0.05, dense_W1, np.zeros_like(dense_W1))
    init_b1 = np.where(abs(dense_b1) > 0.05, dense_b1, np.zeros_like(dense_b1))
    init_W2 = np.where(abs(dense_W2) > 0.05, dense_W2, np.zeros_like(dense_W2))
    init_b2 = np.where(abs(dense_b2) > 0.05, dense_b2, np.zeros_like(dense_b2))

    neurons = np.array([], dtype='int64')
    for i in subset:
        neurons = np.union1d(neurons, np.where(abs(dense_W2[:, i]) > 0.05))

    # for layer1
    init_W1 = init_W1[:, neurons]
    init_b1 = init_b1[neurons]
    L1_output_size = int(init_b1.shape[0])

    # for layer2
    init_W2 = init_W2[neurons, :]
    init_W2 = init_W2[:, subset]
    init_b2 = init_b2[subset]

    # construct SubNet
    submodel = models.Sequential()
    submodel.add(layers.Dense(L1_output_size, activation='relu', input_shape=(784,)))
    submodel.add(layers.Dense(num_subset))
    submodel.add(layers.Activation('softmax'))

    submodel.layers[0].set_weights([init_W1, init_b1])
    submodel.layers[1].set_weights([init_W2, init_b2])
    submodel.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    if meanNodes:
        # for calculate mean of nodes
        A0 = np.where(abs(dense_W2[:, 0]) > 0.05)
        A1 = np.where(abs(dense_W2[:, 1]) > 0.05)
        A2 = np.where(abs(dense_W2[:, 2]) > 0.05)
        A3 = np.where(abs(dense_W2[:, 3]) > 0.05)
        A4 = np.where(abs(dense_W2[:, 4]) > 0.05)
        A5 = np.where(abs(dense_W2[:, 5]) > 0.05)
        A6 = np.where(abs(dense_W2[:, 6]) > 0.05)
        A7 = np.where(abs(dense_W2[:, 7]) > 0.05)
        A8 = np.where(abs(dense_W2[:, 8]) > 0.05)
        A9 = np.where(abs(dense_W2[:, 9]) > 0.05)

        items = [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]

        return submodel, items
    else:
        return submodel, None


def FashionMNIST_SubNet(model, subset, meanNodes=False):
    num_subset = len(subset)
    Conv_W1, Conv_b1 = model.layers[0].get_weights()
    Conv_W2, Conv_b2 = model.layers[3].get_weights()
    dense_W1, dense_b1 = model.layers[7].get_weights()
    dense_W2, dense_b2 = model.layers[9].get_weights()

    dense_init_W1 = np.where(abs(dense_W1) > 0.05, dense_W1, np.zeros_like(dense_W1))
    dense_init_b1 = np.where(abs(dense_b1) > 0.05, dense_b1, np.zeros_like(dense_b1))
    dense_init_W2 = np.where(abs(dense_W2) > 0.05, dense_W2, np.zeros_like(dense_W2))
    dense_init_b2 = np.where(abs(dense_b2) > 0.05, dense_b2, np.zeros_like(dense_b2))

    neurons = np.array([], dtype='int64')
    for i in subset:
        neurons = np.union1d(neurons, np.where(abs(dense_W2[:, i]) > 0.05))

    # for dense layer1
    dense_init_W1 = dense_init_W1[:, neurons]
    dense_init_b1 = dense_init_b1[neurons]
    dense_L1_output_size = int(dense_init_b1.shape[0])

    # for dense layer2
    dense_init_W2 = dense_init_W2[neurons, :]
    dense_init_W2 = dense_init_W2[:, subset]
    dense_init_b2 = dense_init_b2[subset]

    # construct SubNet
    submodel = models.Sequential()
    submodel.add(layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)))
    submodel.add(layers.MaxPooling2D(pool_size=2))
    submodel.add(layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    submodel.add(layers.MaxPooling2D(pool_size=2))
    submodel.add(layers.Flatten())
    submodel.add(layers.Dense(dense_L1_output_size, activation='relu'))
    submodel.add(layers.Dense(num_subset))
    submodel.add(layers.Activation('softmax'))

    submodel.layers[0].set_weights([Conv_W1, Conv_b1])
    submodel.layers[2].set_weights([Conv_W2, Conv_b2])
    submodel.layers[5].set_weights([dense_init_W1, dense_init_b1])
    submodel.layers[6].set_weights([dense_init_W2, dense_init_b2])
    submodel.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    if meanNodes:
        # for calculate mean of nodes
        A0 = np.where(abs(dense_W2[:, 0]) > 0.05)
        A1 = np.where(abs(dense_W2[:, 1]) > 0.05)
        A2 = np.where(abs(dense_W2[:, 2]) > 0.05)
        A3 = np.where(abs(dense_W2[:, 3]) > 0.05)
        A4 = np.where(abs(dense_W2[:, 4]) > 0.05)
        A5 = np.where(abs(dense_W2[:, 5]) > 0.05)
        A6 = np.where(abs(dense_W2[:, 6]) > 0.05)
        A7 = np.where(abs(dense_W2[:, 7]) > 0.05)
        A8 = np.where(abs(dense_W2[:, 8]) > 0.05)
        A9 = np.where(abs(dense_W2[:, 9]) > 0.05)

        items = [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]

        return submodel, items
    else:
        return submodel, None

