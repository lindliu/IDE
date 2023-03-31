#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 12:26:09 2023

@author: dliu
"""


import numpy as np
import cv2

mapp = np.load('../data/mapp.npy')
mapp = mapp*255
mapp = mapp.astype('uint8')

assert mapp.shape[1]<800

zoom = 800/mapp.shape[1]
mapp = np.repeat(mapp, zoom, axis=1)
mapp = np.repeat(mapp, zoom, axis=2)

size = mapp.shape[1], mapp.shape[2]
duration = 60
fps = mapp.shape[0]/duration

out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
for i in range(int(duration*fps)):
    img = mapp[i]#np.random.randint(0, 256, size, dtype='uint8')
    out.write(img)
out.release()