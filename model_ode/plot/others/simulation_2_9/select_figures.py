#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 12:02:37 2023

@author: dliu
"""

import os
import glob
import shutil
import numpy as np

dir_from = 'figures_full_'#'figures_full'
dir_to = 'figures_end_'#'figures_end'

dirs = glob.glob(f'./{dir_from}/*')
dirs_new = [path.replace(f'{dir_from}', f'{dir_to}') for path in dirs]

for i in range(len(dirs_new)):
    os.makedirs(dirs_new[i], exist_ok=True)

# im_p, npz_p = [], []
for i in range(len(dirs)):
    imgs = glob.glob(dirs[i]+'/*.png')
    npzs =  glob.glob(dirs[i]+'/*.npz')
    
    idx_sorted = np.argsort([int(os.path.split(imgs[i])[-1][:-4]) for i in range(len(imgs))])
    imgs = np.array(imgs)[idx_sorted]
    idx_sorted = np.argsort([int(os.path.split(npzs[i])[-1][:-4]) for i in range(len(npzs))])
    npzs = np.array(npzs)[idx_sorted]
    
    ### copy end figures and data to new directory
    shutil.copyfile(imgs[-1], imgs[-1].replace(f'{dir_from}', f'{dir_to}'))
    shutil.copyfile(npzs[-1], npzs[-1].replace(f'{dir_from}', f'{dir_to}'))

    # im_p.append(imgs[-1])
    # npz_p.append(npzs[-1])
