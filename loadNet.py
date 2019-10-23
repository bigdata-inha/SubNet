import os
import numpy as np
from keras import layers
from keras import models
import keras.regularizers as regularizers
from keras.callbacks import ModelCheckpoint
from dataset import MNISTdata
from dataset import FashionMNISTdata


def MNIST_net():
    np.random.seed(0)

    train_images, train_labels, test_images, test_labels = MNISTdata()
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(784,)))
    model.add(layers.Dense(10, kernel_regularizer=regularizers.l1(0.001)))
    model.add(layers.Activation('softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    MODEL_SAVE_FOLDER_PATH = './MNIST_best/'
    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
        os.mkdir(MODEL_SAVE_FOLDER_PATH)
    model_path = MODEL_SAVE_FOLDER_PATH + 'MNIST.model.best.hdf5'

    if os.path.isfile(model_path):
        model.load_weights(model_path)
    else:
        print("Please wait for train Network...")
        checkpointer = ModelCheckpoint(filepath=model_path, verbose = 0, save_best_only=True)
        model.fit(train_images, train_labels, verbose = 0, validation_data=(test_images, test_labels), epochs=100, batch_size=256, callbacks=[checkpointer])
        print("\nTrain Finish!!")

    model.load_weights(model_path)

    return model

def FashionMNIST_net():
    np.random.seed(0)

    train_images, train_labels, test_images, test_labels = FashionMNISTdata()
    model = models.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, kernel_regularizer=regularizers.l1(0.001)))
    model.add(layers.Activation('softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    MODEL_SAVE_FOLDER_PATH = './FashionMNIST_best/'
    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
        os.mkdir(MODEL_SAVE_FOLDER_PATH)
    model_path = MODEL_SAVE_FOLDER_PATH + 'FashionMNIST.model.best.hdf5'

    if os.path.isfile(model_path):
        model.load_weights(model_path)
    else:
        print("Please wait for train Network...")
        checkpointer = ModelCheckpoint(filepath=model_path, verbose=0, save_best_only=True)
        model.fit(train_images, train_labels, verbose = 0, validation_data=(test_images, test_labels), epochs=100, batch_size=256, callbacks=[checkpointer])
        print("\nTrain Finish!!")

    model.load_weights(model_path)

    return model