# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 03:26:05 2019

@author: Wei-Hsiang, Shen
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class Model(object):
    def __init__(self):
        # In practice, these should be initialized to random values.
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b

# MSE Loss
def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))

# Create training data
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs = tf.random.normal(shape=[NUM_EXAMPLES])
noise  = tf.random.normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

# Get our model
model = Model()

# Training step
@tf.function # function decorator (Auto Graph)
def train_step(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW) # -=
    model.b.assign_sub(learning_rate * db)

# Collect the history of W-values and b-values to plot later
Ws, bs, losses = [], [], []
epochs = 20
for epoch in range(epochs):
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(model(inputs), outputs)
    losses.append(current_loss)

    lr = 0.1 - 0.001*epoch

    train_step(model, inputs, outputs, learning_rate=lr)
    print('Epoch {}: W={:.2f} b={:.2f}, loss={:.2f}'.format(epoch+1, Ws[-1], bs[-1], current_loss))

# Plot training progress
plt.figure()
plt.plot(range(epochs), losses)
plt.title("Loss")

plt.figure()
plt.plot(range(epochs), Ws, 'r', range(epochs), bs, 'b')
plt.plot([TRUE_W] * epochs, 'r--', [TRUE_b] * epochs, 'b--')
plt.legend(['W', 'b', 'true W', 'true_b'])
plt.show()