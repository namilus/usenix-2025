import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



def LeNet(input_shape, num_labels):
    """LeNet model architecture from cleverhans/Forging"""
    model = tf.keras.models.Sequential([
            layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape, padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            layers.Flatten(),
            layers.Dense(120, activation='relu'),
            layers.Dense(84, activation='relu'),
            layers.Dense(num_labels)
        ])        

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return model, loss_fn


def VGGmini(input_shape, num_labels):
    """VGG mini model architecture from Baluta et al. 2023

    """
    model = tf.keras.models.Sequential([
        layers.Conv2D(64, (3,3), strides=(1,1), padding="same", activation='elu', input_shape=input_shape),
        layers.Conv2D(64, (3,3), strides=(1,1), padding="same", activation='elu'),
        layers.MaxPooling2D(pool_size=(2,2)),


        layers.Conv2D(128, (3,3), strides=(1,1), padding="same", activation='elu'),
        layers.Conv2D(128, (3,3), strides=(1,1), padding="same", activation='elu'),
        layers.MaxPooling2D(pool_size=(2,2)),


        layers.Conv2D(256, (3,3), strides=(1,1), padding="same", activation='elu'),
        layers.Conv2D(256, (3,3), strides=(1,1), padding="same", activation='elu'),
        layers.MaxPooling2D(pool_size=(2,2)),

        layers.Conv2D(512, (3,3), strides=(1,1), padding="same", activation='elu'),
        layers.Conv2D(512, (3,3), strides=(1,1), padding="same", activation='elu'),


        layers.Flatten(),
        layers.Dense(128, activation='elu'),
        layers.Dense(64, activation='elu'),
        layers.Dense(num_labels)
        ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return model, loss_fn





def FCN_single(input_shape, num_labels=1):
    """ Simple 2 layer nn that outputs a single number"""
    model = tf.keras.models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(100, activation='relu'),
        layers.Dense(1)
    ])

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return model, loss_fn



class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = x.shape[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    


def Transformer(input_shape=(200,), num_labels=2):
    embed_dim = 32
    num_heads = 2
    ff_dim = 32
    maxlen = 200
    vocab_size = 20_000
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(20, activation="relu")(x)
    outputs = layers.Dense(2)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return model, loss_fn
