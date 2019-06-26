# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:57:02 2019

@author: Wei-Hsiang, Shen
"""

import tensorflow as tf
import tensorflow.keras.layers as layers


def Get_Model():
    inputs_stamp_image = tf.keras.Input(shape=(50,50,3), name='stamp_img')
    inputs_address = tf.keras.Input(shape=(None,), name='address')
    inputs_color = tf.keras.Input(shape=(4,), name='color') # categorical inputs should be one-hot coded

    # Stamp (images data)
    features_stamp = layers.Conv2D(8,3)(inputs_stamp_image)
    features_stamp = layers.MaxPool2D()(features_stamp)
    features_stamp = layers.Flatten()(features_stamp)

    # Address (sequence data)
    num_words = 10000
    features_address = layers.Embedding(num_words, 64)(inputs_address)
    features_address = layers.LSTM(128)(features_address)

    # Color (tabular data)
    features_color = layers.Dense(10)(inputs_color)

    # Merge all features into a single vector via concatenation
    features_all = tf.concat([features_stamp, features_address, features_color], axis=-1)

    # Predict price
    price_prediction = layers.Dense(10)(features_all)
    price_prediction = layers.Dense(1, activation='relu', name='price')(price_prediction)

    # Predict class
    class_prediction = layers.Dense(20)(features_all)
    class_prediction = layers.Dense(10, activation='softmax', name='class')(class_prediction)

    # Instantiate an end-to-end model predicting both priority and department
    model = tf.keras.Model(inputs=[inputs_stamp_image, inputs_address, inputs_color],
                           outputs=[price_prediction, class_prediction])

    return model

if __name__ == '__main__':
    model = Get_Model()
    model.summary()
    tf.keras.utils.plot_model(model, 'Post_Office_Network.png', show_shapes=True)

    # We can assign different loss to each output, and also different weights to each loss
    model.compile(optimizer=tf.keras.optimizers.RMSprop(1e-3),
                  loss={'price': 'mse', # specified by output name
                        'class': 'categorical_crossentropy'},
                  loss_weights=[1., 0.2])





