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
dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

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