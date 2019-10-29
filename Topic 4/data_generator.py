# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 22:18:59 2019

@author: Wei-Hsiang, Shen
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds


def Preprocess_and_Augment(image, label):
    image = tf.image.resize(image, (224, 224))
    image = image/255.0

    image = tf.image.random_flip_left_right(image)

    return image, label

def Get_DS(BATCH_SIZE=16):
    datasets, info = tfds.load("rock_paper_scissors", with_info=True, as_supervised=True, data_dir='./data/')

    data_train = datasets['train']
    data_test = datasets['test']

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    data_train = data_train.map(Preprocess_and_Augment, AUTOTUNE)
    data_test = data_test.map(Preprocess_and_Augment, AUTOTUNE)

    data_train = data_train.shuffle(1024).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    data_test = data_test.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    train_count = info.splits['train'].num_examples
    test_count = info.splits['test'].num_examples
    label_name = info.features['label'].int2str

    return data_train, data_test, train_count, test_count, label_name

if __name__ == '__main__':
    data_train, data_test, train_count, test_count, label_name = Get_DS()

    print("Train Count: {}".format(train_count))
    print("Test Count: {}".format(test_count))
    print(data_train)
    print(data_test)

    for batch in data_train.take(1):
        img_batch = batch[0].numpy()
        label_batch = batch[1].numpy()

    i_pic = 0
    plt.imshow(np.squeeze(img_batch[i_pic]))
    plt.title(label_name(label_batch[i_pic]))
    plt.show()