# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 00:37:01 2019

@author: Wei-Hsiang, Shen
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob

output_file = './progress_small.gif'

with imageio.get_writer(output_file, mode='I') as writer:
    images_files = glob.glob('./generated_images/image*.png')
    images_files = sorted(images_files)
    last = -1
    for i, filename in enumerate(images_files):
        frame = 1.5*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)