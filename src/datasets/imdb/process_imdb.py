import tensorflow as tf
import numpy as np

V = 20_000
maxlen = 200
def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=V)
    x_train = tf.keras.utils.pad_sequences(x_train, maxlen=maxlen)
    x_test = tf.keras.utils.pad_sequences(x_test, maxlen=maxlen)

    print(x_train.shape, x_test.shape)
    np.savez("imdb.npz",
             x_train = x_train,
             y_train = y_train,
             x_test = x_test,
             y_test = y_test)
             


if __name__ == "__main__":
    main()
