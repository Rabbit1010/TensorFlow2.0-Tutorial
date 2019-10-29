# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:42:51 2019

@author: Wei-Hsiang, Shen
"""

import random
import matplotlib.pyplot as plt
import csv
import tensorflow as tf


def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.resize(img, [512,512])
    img /= 255.0
    return img

def augmentation(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img

def Get_AOI_DS(BATCH_SIZE=32):
    # Get all data path
    img_path_list = []
    label_list = []
    with open('train.csv', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'ID':
                continue
            img_path_list.append('./train_images/' + row[0])
            label_list.append(int(row[1]))

    # Shuffle the list
    temp = list(zip(img_path_list, label_list))
    random.shuffle(temp)
    img_path_list, label_list = zip(*temp)
    img_path_list, label_list = list(img_path_list), list(label_list)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    DATA_SIZE = len(label_list)

    # Construct dataset
    path_ds = tf.data.Dataset.from_tensor_slices(img_path_list)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    image_ds = image_ds.map(augmentation, num_parallel_calls=AUTOTUNE)

    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(label_list, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    # Train/test split
    train_ds = image_label_ds.take(int(DATA_SIZE*0.8))
    val_ds = image_label_ds.skip(int(DATA_SIZE*0.8))

    train_ds = train_ds.shuffle(buffer_size=DATA_SIZE)
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(AUTOTUNE)

    val_ds = val_ds.repeat()
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, DATA_SIZE

if __name__ == '__main__':
    train_ds, val_ds, _ = Get_AOI_DS()
    print(train_ds)
    print(val_ds)

    for batch in train_ds.take(1):
        pass

    img_batch = batch[0].numpy()
    label_batch = batch[1].numpy()

    i_pic = 15
    plt.imshow(img_batch[i_pic,:,:,0], cmap='gray')
    plt.title(label_batch[i_pic])
    plt.show()

    for batch in val_ds.take(1):
        pass

    img_batch = batch[0].numpy()
    label_batch = batch[1].numpy()

    i_pic = 0
    plt.imshow(img_batch[i_pic,:,:,0], cmap='gray')
    plt.title(label_batch[i_pic])
    plt.show()