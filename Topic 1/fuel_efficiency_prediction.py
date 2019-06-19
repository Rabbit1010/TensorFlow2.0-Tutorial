# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 00:23:11 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import seaborn as sns

# Download Auto MPG dataset
dataset_path = tf.keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(dataset_path)

# Import dataset using pandas read_csv
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()

# Inspect dataset
dataset_arr = np.asarray(dataset)

print(dataset.isna().sum()) # check unknown values
dataset = dataset.dropna() # drop unknown values

# The "Origin" column is really categorical, not numeric. So convert that to a one-hot
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
print(dataset.tail())

# Train/test split
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Split out labels
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# Check the overall statistic
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()

# Preproces the data (normalization)
def Normalize(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = Normalize(train_dataset)
normed_test_data = Normalize(test_dataset)

# Build the model
def Build_Model():
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model

model = Build_Model()

# Inspect the model
model.summary()
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

# Setup training callbacks
#checkpoint_path = "./checkpoints/net2_weights_{epoch:02d}.h5"
#save_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True,
#                                                     period=1)
#csv_logger = tf.keras.callbacks.CSVLogger('./checkpoints/training2.log')

# Train the model
history = model.fit(
    normed_train_data, train_labels,
    epochs=1000, validation_split = 0.2, verbose=2)

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()

plot_history(history)