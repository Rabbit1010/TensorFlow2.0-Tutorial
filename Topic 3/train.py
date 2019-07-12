# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:33:24 2019

@author: Wei-Hsiang, Shen
"""

import tensorflow as tf
from model import AOI_model
from generate_data import Get_AOI_DS


if __name__ == '__main__':
    print("Version: ", tf.__version__)
    print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

    BATCH_SIZE = 32
    EPOCH = 50

    # Get the dataset
    train_ds, val_ds, data_size = Get_AOI_DS(BATCH_SIZE)

    # Initialize the model
    model = AOI_model()

    # Setup call backs
    checkpoint_path = "./checkpoints/resnet50_{epoch:03d}_{val_acc:.5f}.h5"
    save_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True)
    csv_logger = tf.keras.callbacks.CSVLogger('./checkpoints/training.log')

    model.fit(x=train_ds, validation_data=val_ds,
              epochs=EPOCH, verbose=1,
              steps_per_epoch=tf.math.ceil(data_size*0.8/BATCH_SIZE).numpy(),
              validation_steps=tf.math.ceil(data_size*0.2/BATCH_SIZE).numpy(),
              callbacks=[save_checkpoint, csv_logger])

