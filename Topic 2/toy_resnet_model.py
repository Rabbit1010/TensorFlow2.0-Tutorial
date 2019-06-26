# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 23:36:00 2019

@author: Wei-Hsiang, Shen
"""

import tensorflow as tf
from tensorflow.keras import layers


def Toy_ResNet():
    inputs = tf.keras.Input(shape=(32,32,3), name='img')

    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    block_1_output = layers.MaxPooling2D(3)(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_1_output)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    block_2_output = layers.add([x, block_1_output])

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_2_output)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    block_3_output = layers.add([x, block_2_output])

    x = layers.Conv2D(64, 3, activation='relu')(block_3_output)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name='toy_resnet')

    return model

if __name__ == '__main__':
    """model testing"""
    model = Toy_ResNet()

    # Plot and inspect the model
    model.summary()
    tf.keras.utils.plot_model(model, 'Toy_ResNet.png', show_shapes=True)










