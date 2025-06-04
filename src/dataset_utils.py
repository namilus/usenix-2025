import random
import tensorflow as tf
from tensorflow import keras
import numpy as np


def load_and_process_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train / 255 ; x_test = x_test / 255
    x_train = tf.expand_dims(x_train, axis=-1)
    x_test = tf.expand_dims(x_test, axis=-1)
    return (x_train, y_train), (x_test, y_test)

def load_and_process_cifar10():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = tf.constant(x_train / 255.0) ; x_test = tf.constant(x_test / 255.0)
    y_train = tf.squeeze(tf.transpose(y_train)); y_test = tf.squeeze(tf.transpose(y_test))
    return (x_train, y_train), (x_test, y_test)


def load_and_process_adult():
    path = "./datasets/adult/adult.npz"
    a = np.load(path)
    return (a['x_train'], a['y_train']), (a['x_test'], a['y_test'])


def load_and_process_imdb():
    path = "./datasets/imdb/imdb.npz"
    a = np.load(path)
    return (a['x_train'], a['y_train']), (a['x_test'], a['y_test'])

