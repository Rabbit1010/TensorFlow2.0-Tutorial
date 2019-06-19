# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 00:52:43 2019

@author: Wei-Hsiang, Shen
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from tensorflow.keras import layers

# Build the model
model = tf.keras.Sequential()
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Load trained model weights
model.load_weights('./checkpoints/net_weights_05.h5')

pic = cv2.imread('./test8.png')

# preprocess
pic = pic / 255.0
pic = np.mean(pic, axis=2)
pic = np.expand_dims(pic, axis=0)

out = model(pic)
out = out.numpy() # float 32

plt.bar([0,1,2,3,4,5,6,7,8,9], np.squeeze(out))
plt.show()