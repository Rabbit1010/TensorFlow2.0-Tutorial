# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 02:51:44 2019

@author: Wei-Hsiang, Shen
"""

import tensorflow as tf
from model import Transferred_MobileNetV2, New_MobileNetV2
from data_generator import Get_DS


if __name__ == '__main__':
    BATCH_SIZE = 16
    LEARNING_RATE = 1e4

    data_train, data_test, train_count, test_count, label_name = Get_DS(BATCH_SIZE)

    model = Transferred_MobileNetV2()
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
#    model.summary()

    # Setup call backs
    checkpoint_path = "./checkpoints/MobileNetV2_transferred_{epoch:03d}.h5"
    save_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True)
    csv_logger = tf.keras.callbacks.CSVLogger('./checkpoints/training.log')

    history = model.fit(x=data_train,
                        epochs=3, verbose=1,
                        steps_per_epoch=round(train_count/BATCH_SIZE),
                        validation_steps=round(test_count/BATCH_SIZE),
                        validation_data=data_test,
                        callbacks=[save_checkpoint, csv_logger])

    # Fine-tuning
    model.trainable = True
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=LEARNING_RATE/10),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
#    model.summary()

    # Setup call backs
    checkpoint_path = "./checkpoints/MobileNetV2_fine_tune_{epoch:03d}.h5"
    save_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True)
    csv_logger = tf.keras.callbacks.CSVLogger('./checkpoints/training_fine_tune.log')

    history = model.fit(x=data_train,
                        epochs=10, verbose=1,
                        steps_per_epoch=round(train_count/BATCH_SIZE),
                        validation_steps=round(test_count/BATCH_SIZE),
                        validation_data=data_test,
                        callbacks=[save_checkpoint, csv_logger])