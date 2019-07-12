# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:14:14 2019

@author: Wei-Hsiang, Shen
"""

import tensorflow as tf
from tensorflow.keras import layers
from resblock import ResBlockBottleneck

def AOI_model():
    inputs_image = tf.keras.Input(shape=(512, 512, 1), name='input_image')
    x = inputs_image

    x = layers.Conv2D(32, 3, strides=(2,2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(64, 3, strides=(2,2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D()(x)

    for _ in range(2):
        x = ResBlockBottleneck(64, 64, 256)(x)
    x = ResBlockBottleneck(64, 64, 256, strides=(2,2))(x)

    for _ in range(3):
        x = ResBlockBottleneck(128, 128, 512)(x)
    x = ResBlockBottleneck(128, 128, 512, strides=(2,2))(x)

    for _ in range(5):
        x = ResBlockBottleneck(256, 256, 1024)(x)
    x = ResBlockBottleneck(256, 256, 1024, strides=(2,2))(x)

    for _ in range(2):
        x = ResBlockBottleneck(512, 512, 2048)(x)
    x = ResBlockBottleneck(512, 512, 2048, strides=(2,2))(x)

    x = layers.AvgPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dense(6, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs_image, outputs=x)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['acc'])

    return model

if __name__ == '__main__':
    model = AOI_model()
    model.summary()
    tf.keras.utils.plot_model(model, 'AOI_model.png', show_shapes=True)