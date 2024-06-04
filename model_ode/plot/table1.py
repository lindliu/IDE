#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 08:44:40 2023

@author: dliu
"""

import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pandas as pd

import matplotlib
font = {#'family' : 'normal',
        # 'weight' : 'normal', #'bold'
        'size'   : 16}
matplotlib.rc('font', **font)

def get_train_data(data, start, length, recovery_time, estimate=True, prop=True, scale=1, data_type='cases_mean'):
    """data_type: 'daily_cases or cases_mean"""
    if estimate==True:
        if prop==True:
            cases_convolved = np.convolve(recovery_time*[1], data['inf_mean'], mode='same')[start:start+length].reshape([1,-1,1])
            data_ = cases_convolved / data['population'].iloc[0]
        else:
            cases_convolved = np.convolve(recovery_time*[1], data['inf_mean'], mode='same')[start:start+length].reshape([1,-1,1])
            data_ = cases_convolved
        
    else:
        if prop==True:
            cases_convolved = np.convolve(recovery_time*[1], data[data_type], mode='same')[start:start+length].reshape([1,-1,1]) * scale
            data_ = cases_convolved / data['population'].iloc[0]
        else:
            cases_convolved = np.convolve(recovery_time*[1], data[data_type], mode='same')[start:start+length].reshape([1,-1,1]) * scale
            data_ = cases_convolved
            
    return data_

def load_result_path(country, start, estimate, prop, length):
    ### load data
    if country!='simulation':
        data = pd.read_csv(f'../../data/covid_{country}.csv', sep='\t')
        data['date'] = pd.to_datetime(data['date'])
        
        data_train = get_train_data(data, start, length=length, recovery_time=14, estimate=estimate, prop=prop)
        data_train = data_train.flatten()
    elif country=='simulation': 
        data = pd.DataFrame(np.load('../../data/simulation_2_3.npy'), columns=['S','I','R'])       
        data['date'] = np.arange(500)
         
        data = data.iloc[start:start+length,:]
        data_train = data['I']
    
    if prop or country=='simulation':
        N = 1
    else:
        N =  int(data['population'].iloc[0])
    
    if country == 'simulation':
        file_name = f'{country}'
    elif estimate:
        file_name = f'estimate_{country}'
    else:
        file_name = f'real_{country}'
                
        
    path_u = glob.glob(f'../results/{file_name}_{start}_*')
    end_idx_ = np.array([int(os.path.split(path_u[i])[-1].split('_')[-1]) for i in range(len(path_u))])
    
    ### data np.arange(20,350,3) 
    mask = np.array([i in start+np.arange(20,350,3) for i in end_idx_])
    idx_sorted = np.argsort(end_idx_[mask])
    path_ = np.array(path_u)[mask][idx_sorted]
    
    ### whole data
    # idx_sorted = np.argsort(end_idx_)
    # path_ = np.array(path_u)[idx_sorted]
    
    path = []
    for p in path_:
        path_s = glob.glob(os.path.join(p, '*.npz'))
        idx_sorted = np.argsort([int(os.path.split(path_s[i])[-1][:-4]) for i in range(len(path_s))])
        path_s = np.array(path_s)[idx_sorted]
        
        path.append(path_s[-1])
        
        
    data_pred = np.load(path[0])
    print(list(data_pred.keys()))
    
    time_day = data['date'][start:start+length]
    return path, data_train, time_day, N, file_name
    

countries = ['Mexico', 'South Africa', 'Republic of Korea', \
             'Mexico', 'South Africa', 'Republic of Korea', 'simulation']  #
start_list = [640, 640, 640, 640, 640, 640, 0]
length = 400

point_show = np.arange(20,350,6)

## first decend
points1 = np.array([71, 41, 134, 62, 31, 120, 39]) ### first peak
points2 = np.array([247, 187, 283, 240, 177, 273, 178]) ### second peak

# ## first decend
# points1 = np.array([75, 45, 140, 65, 35, 125, 45])
# points2 = points1+40

# ## second decend
# points1 = np.array([250, 190, 290, 245, 180, 280, 185])
# points2 = points1+40

fig, ax = plt.subplots(3,3,figsize=[30,20])
ax = ax.flatten()
fig1, ax1 = plt.subplots(3,3,figsize=[30,20])
ax1 = ax1.flatten()
for idx in range(len(countries)):
    country = countries[idx]
    start = start_list[idx]
    
    estimate, prop = (False, False) if idx<=2 else (True, True)
    if country=='simulation':
        filename = 'simulation'
    else:
        filename = f'estimate_{country}' if estimate else f'real_{country}'
    
    path, data_train, time_day, N, file_name = load_result_path(country, start, estimate, prop, length)
    pred_length = 7*4
    pred_idx, prediction_I, data_true = [], [], []
    prediction_S, prediction_R = [], []
    mu_list, sigma_list = [], []
    tau_list = []
    
    t_end = length/14
    
    ### all data result
    for pp in path:
        
        idx_end = int(pp.split('/')[-2].split('_')[-1])
        pos = idx_end-start
    
        data_pred = np.load(pp)
        pred = data_pred['pred'][:,:length,:]
        
        if pos in point_show:
            mu_list.append(data_pred['mu'].item())
            sigma_list.append(data_pred['sigma'].item())
            tau_list.append(data_pred['tau'].item())
            
            pred_idx.append(list(np.arange(pos, pos+pred_length))[-1])
            prediction_I.append(list(pred[0,pos:pos+pred_length,1]))
            data_true.append(data_train[pos:pos+pred_length])
            
            
            prediction_S.append(list(pred[0,pos:pos+pred_length,0])[-1])
            prediction_R.append(list(pred[0,pos:pos+pred_length,2])[-1])
    
    
    
    ### part data result
    # for pp in path:
        
    #     idx_end = int(pp.split('/')[-2].split('_')[-1])
    #     pos = idx_end-start

    #     if pos>=points1[idx] and pos<=points2[idx]:
    #         data_pred = np.load(pp)
    #         pred = data_pred['pred'][:,:length,:]
            
    #         mu_list.append(data_pred['mu'].item())
    #         sigma_list.append(data_pred['sigma'].item())
    #         tau_list.append(data_pred['tau'].item())
            
    #         pred_idx.append(list(np.arange(pos, pos+pred_length))[-1])
    #         prediction_I.append(list(pred[0,pos:pos+pred_length,1]))
    #         data_true.append(data_train[pos:pos+pred_length])
            
            
    #         prediction_S.append(list(pred[0,pos:pos+pred_length,0])[-1])
    #         prediction_R.append(list(pred[0,pos:pos+pred_length,2])[-1])
    
    mu_list = np.array(mu_list)
    sigma_list = np.array(sigma_list)
    
    data_true = np.array(data_true)
    
    prediction_I = np.array(prediction_I)
    prediction_S = np.array(prediction_S)
    prediction_R = np.array(prediction_R)
    
    
    error = abs(data_true-prediction_I)/data_train.max()  ### relative error
    # error = abs(data_true-prediction_I)/data_true  ### error
    
    error_1 = np.median(error[:,:7], axis=1).mean()
    error_2 = np.median(error[:,:14], axis=1).mean()
    error_3 = np.median(error[:,:21], axis=1).mean()
    error_4 = np.median(error[:,:28], axis=1).mean()
    error_m_week = np.array([error_1,error_2,error_3,error_4])
    print(f'MARPE. {filename}: {error_m_week.astype(np.float16)}')
    
    
    error = abs(data_true-prediction_I)/data_true  ### error
    error_1 = np.median(error[:,:7], axis=1).mean()
    error_2 = np.median(error[:,:14], axis=1).mean()
    error_3 = np.median(error[:,:21], axis=1).mean()
    error_4 = np.median(error[:,:28], axis=1).mean()
    error_m_week = np.array([error_1,error_2,error_3,error_4])
    print(f'MAPE. {filename}: {error_m_week.astype(np.float16)}')
    
    
    # ax[idx].plot(error_m)
    ax[idx].plot(np.repeat(error_m_week,7))
    ax[idx].set_title(f'{filename}')


    ax1[idx].plot(data_train)
    ax1[idx].axvline(x=points1[idx], color='k', linestyle='dashed', label='axvline')
    ax1[idx].axvline(x=points2[idx], color='r', linestyle='dashed', label='axvline')
