# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 19:28:06 2019

@author: Wei-Hsiang, Shen
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers

# Check version
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

# Download MINST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Let's plot some image
plt.imshow(train_images[0,:,:], cmap='gray')
plt.colorbar()
plt.title(train_labels[0])
plt.show()

plt.imshow(test_images[50,:,:], cmap='gray')
plt.colorbar()
plt.title(test_labels[50])
plt.show()

# Preprocess the data
# For float16 training: https://arxiv.org/abs/1502.02551
train_images = train_images / 255.0 # float32 is standard
test_images = test_images / 255.0

#train_images = np.expand_dims(train_images, axis=-1)
#test_images = np.expand_dims(test_images, axis=-1)

# Build the model
model = tf.keras.Sequential()
#model.add(layers.Conv2D(filters=4, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Inspect model (check the parameters)
model.summary()
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Set-up callbacks
checkpoint_path = "./checkpoints/net_weights_{epoch:02d}.h5"
save_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True,
                                                     period=1)
csv_logger = tf.keras.callbacks.CSVLogger('./checkpoints/training.log')


# Train the model
# https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model#fit
model.fit(x=train_images,
          y=train_labels,
          batch_size=32,
          epochs=10,
          callbacks=[save_checkpoint, csv_logger])

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy:', test_acc)

# Make predictions
#img = test_images[100,:,:]
#plt.imshow(img, cmap='gray')
#plt.colorbar()
#plt.title(test_labels[100])
#plt.show()
#
#predictions = model(np.expand_dims(img, axis=0))
#predictions = model.predict(np.expand_dims(img, axis=0))
#predicted_class = np.argmax(predictions)