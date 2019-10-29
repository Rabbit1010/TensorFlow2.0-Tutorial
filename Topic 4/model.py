# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 02:13:59 2019

@author: Wei-Hsiang, Shen
"""

import tensorflow as tf


def Transferred_MobileNetV2():
    # Download MobileNet-V2 with pretrained weight on ImageNet
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
#    model.summary()

    # Trim off the last FC layer
    base_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    base_model.trainable = False # Freeze the convolutional base
#    base_model.summary()

    # Reconstruct the FC layer using functional API
    # print(model.layers[-1].activation)
    x = base_model(model.inputs)
    x = tf.keras.layers.Dense(3, activation='softmax')(x)
    new_model = tf.keras.Model(inputs=model.inputs, outputs=x)
#    new_model.summary()

    # Or we can use the simpler sequential API
    fc_layer = tf.keras.layers.Dense(3, activation='softmax')
    new_model = tf.keras.Sequential([
            base_model,
            fc_layer])

    return new_model

def New_MobileNetV2():
    # Download MobileNet-V2 with pretrained weight on ImageNet
    model = tf.keras.applications.MobileNetV2(weights='imagenet')

    # Trim off the last FC layer
    base_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)

    # Add the lasy FC layer
    fc_layer = tf.keras.layers.Dense(3, activation='softmax')
    new_model = tf.keras.Sequential([
            base_model,
            fc_layer])

    new_model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    return new_model

if __name__ == '__main__':
    model = Transferred_MobileNetV2()
    model.summary()

#    model = New_MobileNetV2()
#    model.summary()