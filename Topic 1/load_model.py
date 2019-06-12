# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 00:52:43 2019

@author: Wei-Hsiang, Shen
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers

# Build the model
model = tf.keras.Sequential()
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Load trained model weights
model.load_weights('./checkpoints/net_weights_010')