# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Fri Apr 28 13:26:25 2023

# @author: dliu
# """


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

def load_train_data(country, start, estimate, prop, length):
    ### load data
    if country!='simulation':
        data = pd.read_csv(f'../data/covid_{country}.csv', sep='\t')
        data['date'] = pd.to_datetime(data['date'])
        
        data_train = get_train_data(data, start, length=length, recovery_time=14, estimate=estimate, prop=prop)
        data_train = data_train.flatten()
    elif country=='simulation': 
        data = pd.DataFrame(np.load('../data/simulation_2_3.npy'), columns=['S','I','R'])       
        data['date'] = np.arange(500)
         
        data = data.iloc[start:start+length,:]
        data_train = data['I']
    
    time_day = data['date'][start:start+length]    
    
    if prop or country=='simulation':
        N = 1
    else:
        N =  int(data['population'].iloc[0])

    return data_train, time_day, N

countries = ['Mexico', 'South Africa', 'Republic of Korea']  #'simulation'
start_list = [640, 640, 640]

i = 0
country = countries[i]
start = start_list[i]

### set estimate=false if using real cases to train
estimate, prop = True, True
# estimate, prop = False, False
length = 400

data_train, time_day, N = load_train_data(country, start, estimate, prop, length)

path_peak_1 = glob.glob('./figures/estimate_Mexico_640_*_first_peak/*.npz')[0]
data_peak_1 = np.load(path_peak_1)
pred_peak_1 = data_peak_1['pred'][0]
beta_peak_1 = data_peak_1['beta']

endind_peak_1 = int(path_peak_1.split('/')[-2].split('_')[-3])
pos_peak_1 = endind_peak_1-start

path_peak_2 = glob.glob('./figures/estimate_Mexico_640_*_second_peak/*.npz')[0]
data_peak_2 = np.load(path_peak_2)
pred_peak_2 = data_peak_2['pred'][0]
beta_peak_2 = data_peak_2['beta']

endind_peak_2 = int(path_peak_2.split('/')[-2].split('_')[-3])
pos_peak_2 = endind_peak_2-start


fig, ax = plt.subplots(1,3, figsize=[26,6])
ax[0].plot(time_day, data_train, c='r', label='estimated I')
ax[0].plot(time_day.iloc[:pos_peak_2], pred_peak_1[:pos_peak_2,1], c='b', linestyle='dashdot', label='predict I (1st peak)')
ax[0].plot(time_day, pred_peak_2[:,1], c='darkgreen', linestyle='dotted', label='predict I (2nd peak)')

ax[0].legend()
ax[0].axvline(x=time_day[endind_peak_1], color='k', linestyle='dashed', label='axvline')
ax[0].axvline(x=time_day[endind_peak_2], color='k', linestyle='dashed', label='axvline')

ax[0].axvspan(time_day.iloc[pos_peak_1], time_day.iloc[pos_peak_2], alpha=0.15, color='b')
ax[0].axvspan(time_day.iloc[pos_peak_2], time_day.iloc[-1], alpha=0.15, color='g')
plt.setp(ax[0].get_xticklabels(), rotation=45)


ax[1].plot(time_day.iloc[:pos_peak_2], pred_peak_1[:pos_peak_2, 0], c='b', label='S (1st peak)')
ax[1].plot(time_day.iloc[:pos_peak_2], pred_peak_1[:pos_peak_2, 2], c='b', linestyle='dashed', label='R (1st peak)')

ax[1].plot(time_day, pred_peak_2[:, 0], c='darkgreen', label='S (2nd peak)')
ax[1].plot(time_day, pred_peak_2[:, 2], c='darkgreen', linestyle='dashed', label='R (2nd peak)')

ax[1].legend()
ax[1].axvline(x=time_day[endind_peak_1], color='k', linestyle='dashed', label='axvline')
ax[1].axvline(x=time_day[endind_peak_2], color='k', linestyle='dashed', label='axvline')

ax[1].axvspan(time_day.iloc[pos_peak_1], time_day.iloc[pos_peak_2], alpha=0.15, color='b')
ax[1].axvspan(time_day.iloc[pos_peak_2], time_day.iloc[-1], alpha=0.15, color='g')
plt.setp(ax[1].get_xticklabels(), rotation=45)


ax[2].plot(time_day.iloc[:pos_peak_2], beta_peak_1[:pos_peak_2, 0], c='b', label='$R_0$ (1st peak)')
ax[2].plot(time_day, beta_peak_2[:, 0], c='darkgreen', label='$R_0$ (2nd peak)')

ax[2].legend()
ax[2].axvline(x=time_day[endind_peak_1], color='k', linestyle='dashed', label='axvline')
ax[2].axvline(x=time_day[endind_peak_2], color='k', linestyle='dashed', label='axvline')

ax[2].axvspan(time_day.iloc[pos_peak_1], time_day.iloc[pos_peak_2], alpha=0.15, color='b')
ax[2].axvspan(time_day.iloc[pos_peak_2], time_day.iloc[-1], alpha=0.15, color='g')
plt.setp(ax[2].get_xticklabels(), rotation=45)



fig.tight_layout()
fig.savefig(f'./figures/prediction_trend.png', \
            bbox_inches='tight', dpi=300)
