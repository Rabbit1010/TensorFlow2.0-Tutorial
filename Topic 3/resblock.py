# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:27:10 2019

@author: user
"""

import tensorflow as tf
from tensorflow.keras import layers


class ResBlockBottleneck(tf.keras.Model):
    """ResBlock (Bottleneck version 3 blocks)"""
    def __init__(self, num_feature_in, num_feature_mid, num_feature_out, strides=(1,1)):
        super(ResBlockBottleneck, self).__init__(name='')
        self.strides = strides
        self.num_feature_in = num_feature_in
        self.num_feature_out = num_feature_out

        self.conv2a = layers.Conv2D(num_feature_mid, 1, use_bias=False)
        self.bn2a = layers.BatchNormalization()

        self.conv2b = layers.Conv2D(num_feature_mid, 3, strides=strides, use_bias=False, padding='same')
        self.bn2b = layers.BatchNormalization()

        self.conv2c = layers.Conv2D(num_feature_out, 1, use_bias=False)
        self.bn2c = layers.BatchNormalization()

        self.conv2d = layers.Conv2D(num_feature_out, 1, use_bias=False, strides=strides)
        self.bn2d = layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        # projection shortcut is used when input and output are different dimensions
        if self.num_feature_in!=self.num_feature_out or self.strides != (1, 1):
            shortcut = self.conv2d(input_tensor)
            shortcut = self.bn2d(shortcut, training=training)
        else: # Identity connection
            shortcut = input_tensor

        x = layers.add([x, shortcut])
        x = tf.nn.relu(x)
        return x