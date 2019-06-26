# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:05:26 2019

@author: Wei-Hsiang, Shen
"""

import tensorflow as tf
from tensorflow.keras import layers


class Conv_Batch_ReLu(object):
    """Convolution2D + Batch Normalization + ReLu is a common combination of layers"""
    def __init__(self, num_features, strides=(1,1)):
        self.num_features = num_features
        self.strides = strides

    def __call__(self, x):
        x = layers.Conv2D(self.num_features, kernel_size=(3,3), strides=self.strides, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

def Low_Level_Features_Network():
    """We build the sub-model so when we call it twice, the weights is shared"""
    inputs = tf.keras.Input(shape=(224, 224, 1))
    x = Conv_Batch_ReLu(64, (2,2))(inputs)
    x = Conv_Batch_ReLu(128)(x)
    x = Conv_Batch_ReLu(128, (2,2))(x)
    x = Conv_Batch_ReLu(256)(x)
    x = Conv_Batch_ReLu(256, (2,2))(x)
    outputs = Conv_Batch_ReLu(512)(x)
    return tf.keras.Model(inputs, outputs, name='Low_Level_Features_Network')

def Mid_Level_Features_Network(x):
    x = Conv_Batch_ReLu(512)(x)
    x = Conv_Batch_ReLu(256)(x)
    return x

def Colorization_Network(x):
    x = Conv_Batch_ReLu(128)(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = Conv_Batch_ReLu(64)(x)
    x = Conv_Batch_ReLu(64)(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = Conv_Batch_ReLu(32)(x)
    x = Conv_Batch_ReLu(2)(x)
    return x

def Global_Features_Network(x):
    x = Conv_Batch_ReLu(512, (2,2))(x)
    x = Conv_Batch_ReLu(512)(x)
    x = Conv_Batch_ReLu(512, (2,2))(x)
    x = Conv_Batch_ReLu(512)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = layers.Dense(512)(x)
    x = layers.Dense(256)(x)
    return x

def Classification_Network(x):
    x = layers.Dense(256)(x)
    x = layers.Dense(205)(x)
    return x

def Automatic_Colorization_Network():
    """
    take in an gray imgae and transform it to color
    Iput (256,)
    """
    # A model can have more than one input
    inputs_luminance = tf.keras.Input(shape=(224,224,1), name='fixed_size_img')
    inputs_luminance_arbitrary = tf.keras.Input(shape=(448,448,1), name='arbitrary_size_img')

    shared_Low_Level_Features_Network = Low_Level_Features_Network()

    # Network path of local features (upper part of the figure)
    x = shared_Low_Level_Features_Network(inputs_luminance_arbitrary)
    local_features = Mid_Level_Features_Network(x)

    # Network path of global features (lower part of the figure)
    y = shared_Low_Level_Features_Network(inputs_luminance)
    global_features = Global_Features_Network(y)
    labels = Classification_Network(global_features)

    # fuse features
    print(global_features.shape)
    z = tf.expand_dims(global_features, axis=1)
    z = tf.expand_dims(z, axis=1)
    print(z.shape)
    z = tf.tile(z, [1,56,56,1])
    print(z.shape)

    fused_features = tf.concat([local_features, z], axis=-1)
    fused_features = Conv_Batch_ReLu(256)(fused_features)

    chrominance = Colorization_Network(fused_features)
    chrominance = layers.UpSampling2D(size=(2, 2))(chrominance)

    outputs_LAB_color = tf.concat([inputs_luminance_arbitrary, chrominance], axis=-1)

    model = tf.keras.Model(inputs=[inputs_luminance_arbitrary, inputs_luminance], outputs=[outputs_LAB_color, labels], name='Automatic_Colorizer')

    return model

if __name__ == '__main__':
    """model testing"""
    model = Automatic_Colorization_Network()

    # Plot and inspect the model
    model.summary()
    tf.keras.utils.plot_model(model, 'Colorization_Network.png', show_shapes=True)