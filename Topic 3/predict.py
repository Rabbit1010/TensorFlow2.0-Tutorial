# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:26:05 2019

@author: Wei-Hsiang, Shen
"""

import numpy as np
import csv
import cv2
from tqdm import tqdm
from model import AOI_model


if __name__ == '__main__':
    # Get all data path
    img_path_list = []
    with open('test.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == 'ID':
                continue
            img_path_list.append('./test_images/' + row[0])

    # Load model
    model = AOI_model()
    model.load_weights('./checkpoints/resnet50_300epoch.h5')

    # Model inference
    predicted_label_list = []
    for i_path in tqdm(img_path_list):
        img = cv2.imread(i_path, cv2.IMREAD_GRAYSCALE)
        img = img / 255.0

        # Create image batch for test time augmentation
        img_batch = np.zeros((4,512,512,1), dtype='float32')
        img_batch[0,:,:,0] = img
        img_batch[1,:,:,0] = img[::-1,:]
        img_batch[2,:,:,0] = img[::-1,::-1]
        img_batch[3,:,:,0] = img[:,::-1]

        prediction = model.predict(img_batch)
        prediction_all = np.sum(prediction, axis=0)
        predicted_label = np.argmax(prediction_all)

        predicted_label_list.append(predicted_label)

    # Write result to .csv file
    assert len(img_path_list) == len(predicted_label_list)

    with open('test_output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'Label'])
        for path, prediction in zip(img_path_list, predicted_label_list):
            file_name = path.split('/')[-1]
            writer.writerow([file_name, prediction])