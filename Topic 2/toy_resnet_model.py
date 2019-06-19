# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 23:36:00 2019

@author: Wei-Hsiang, Shen
"""

import tensorflow as tf
from tensorflow.keras import layers


def Toy_ResNet():
    inputs = tf.keras.Input(shape=(32,32,3), name='img')
    x = inputs

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

def Load_and_Preprocess_Data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return X_train, X_test, y_train, y_test

# Get model
net = Toy_ResNet()

# Get data
X_train, X_test, y_train, y_test = Load_and_Preprocess_Data()

# Plot and inspect the model
net.summary()
tf.keras.utils.plot_model(net, 'Toy_ResNet.png', show_shapes=True)

# Compile the model
net.compile(optimizer=tf.keras.optimizers.RMSprop(1e-3),
              loss='categorical_crossentropy',
              metrics=['acc'])

# Train the model
net.fit(X_train, y_train,
          batch_size=64,
          epochs=1,
          validation_split=0.2)