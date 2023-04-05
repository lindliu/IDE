#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 10:19:57 2023

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

i = 3
start_list = [0, 640, 640, 640]
countries = ['simulation', 'Mexico', 'South Africa', 'Republic of Korea']
country = countries[i]
start = start_list[i]

### set estimate=false if using real cases to train
# estimate, prop = True, True 
estimate, prop = False, False 

length = 400
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

if prop:
    N = 1
else:
    N =  int(data['population'].iloc[0])

if country == 'simulation':
    file_name = f'{country}'
elif estimate:
    file_name = f'estimate_{country}'
else:
    file_name = f'real_{country}'
            
    
path_u = glob.glob(f'./figures/{file_name}_{start}_*')

idx_sorted = np.argsort([int(os.path.split(path_u[i])[-1].split('_')[-1]) for i in range(len(path_u))])
path_ = np.array(path_u)[idx_sorted]

path = []
for p in path_:
    path_s = glob.glob(os.path.join(p, '*.npz'))
    idx_sorted = np.argsort([int(os.path.split(path_s[i])[-1][:-4]) for i in range(len(path_s))])
    path_s = np.array(path_s)[idx_sorted]
    
    path.append(path_s[-1])
    
    
data_pred = np.load(path[0])
print(list(data_pred.keys()))

time_day = data['date'][start:start+length]

pred_length, pred_length_ = 2, 7
pred_idx, prediction_I, prediction_I_ = [], [], []
prediction_S, prediction_R = [], []
mu_list, sigma_list = [], []
tau_list = []

t_end = 25
T = np.linspace(0., t_end, length)[::-1]
                   
for pp in path:
    
    idx_end = int(pp.split('/')[-2].split('_')[-1])
    pos = idx_end-start

    data_pred = np.load(pp)
    
    if length>time_day.shape[0]:
        pred = data_pred['pred'][:,:time_day.shape[0],:]
    else:
        pred = data_pred['pred']
        
    mu_list.append(data_pred['mu'].item())
    sigma_list.append(data_pred['sigma'].item())
    tau_list.append(data_pred['tau'].item())
    # pred_idx.extend(list(np.arange(idx_end-start, idx_end-start+pred_length)))
    # prediction_I.extend(list(pred[0,idx_end-start:idx_end-start+pred_length,1]))
    
    pred_idx.append(list(np.arange(pos, pos+pred_length))[-1])
    prediction_I.append(list(pred[0,pos:pos+pred_length,1])[-1])
    prediction_I_.append(list(pred[0,pos:pos+pred_length_,1])[-1])
    
    prediction_S.append(list(pred[0,pos:pos+pred_length,0])[-1])
    prediction_R.append(list(pred[0,pos:pos+pred_length,2])[-1])
    
    # ax[3].scatter(time_day.iloc[np.arange(idx_end-start, idx_end-start+pred_length)], \
    #             pred[0,idx_end-start:idx_end-start+pred_length,1], s=1)

mu_list = np.array(mu_list)
sigma_list = np.array(sigma_list)

prediction_I = np.array(prediction_I)
prediction_I_ = np.array(prediction_I_)
prediction_S = np.array(prediction_S)
prediction_R = np.array(prediction_R)

if country=='simulation': 
    fig, ax = plt.subplots(2,3,figsize=(20,9))
    ax = ax.flatten()

    ax[0].plot(time_day, data_train, c='r', label='Synthetic I')
    ax[0].scatter(time_day.iloc[pred_idx], prediction_I, s=20, c='tab:blue', label=f'{pred_length} days predict I')
    ax[0].scatter(time_day.iloc[pred_idx], prediction_I_, s=20, facecolors='none', edgecolors='tab:green', label=f'{pred_length_} days predict I')
    ax[0].legend()
    # plt.setp(ax[0].get_xticklabels(), rotation=45)
    ax[0].set_title(f"(a)")
    
    ax[1].plot(time_day.iloc[pred_idx], mu_list, linestyle='dashed', marker='o', label='$\mu$')
    ax[1].legend()
    n = 3 ## how many sigmas
    ax[1].fill_between(time_day.iloc[pred_idx], np.clip(mu_list-sigma_list*n,0,1000), mu_list+sigma_list*n,
        alpha=0.2, facecolor='#089FFF', #edgecolor='#1B2ACC', linewidth=1, 
        linestyle='dashdot', antialiased=True, label='3$\sigma$')
    ax[1].legend()
    ax[1].axhline(y=81.1, color='k', linestyle='dashed', label='axvline')
    extraticks = [81.1]
    ax[1].set_yticks(list(ax[1].get_yticks())[1:-1] + extraticks)
    # plt.setp(ax[1].get_xticklabels(), rotation=45)
    ax[1].set_title(f"(b)")


    # l = int(len(path) - len(path)//2.3)
    l = 21
    pp = path[l]
    idx_end = int(pp.split('/')[-2].split('_')[-1])
    pos = idx_end-start
    data_pred = np.load(pp)
    pred = data_pred['pred']
    
    S = data['S'][start:start+length].to_numpy()
    R = data['R'][start:start+length].to_numpy()

    ax[2].plot(time_day.iloc[:pos], data_train[:pos], c='tab:orange', label='train I')
    ax[2].plot(time_day.iloc[pos:], data_train[pos:], c='r', label='test I')
    ax[2].plot(time_day, data_pred['pred'][0,:,1], c='tab:blue', linestyle='dashed', label='predict I')
    # ax[2].plot(time_day.iloc[pos:], data_pred['pred'][0,pos:,1], c='r', linestyle='dashed', label='predict I')
    ax[2].legend()
    ax[2].axvline(x=time_day[idx_end], color='k', linestyle='dashed', label='axvline')
    extraticks = [time_day[idx_end]]
    ax[2].set_xticks(list(ax[2].get_xticks())[1:-1] + extraticks)
    # plt.setp(ax[2].get_xticklabels(), rotation=45)
    ax[2].set_title(f"(c)")
    
    ax[3].plot(time_day, S, c='r', label='S')
    ax[3].plot(time_day, pred[0,:,0], c='tab:blue', linestyle='dashdot', label='predict S')
    # ax[3].plot(time_day.iloc[0:pos], pred[0,:pos,0], c='tab:green', linestyle='dashdot', label='S')
    # ax[3].plot(time_day.iloc[pos:length], pred[0,pos:,0], c='r', linestyle='dashdot', label='predict S')
    ax[3].legend()
    ax[3].axvline(x=time_day[idx_end], color='k', linestyle='dashed', label='axvline')
    extraticks = [time_day[idx_end]]
    ax[3].set_xticks(list(ax[3].get_xticks())[1:-1] + extraticks)
    # plt.setp(ax[3].get_xticklabels(), rotation=45)
    ax[3].set_title(f"(d)")
    
    ax[4].plot(time_day, R, c='r', label='R')
    ax[4].plot(time_day, pred[0,:,2], c='tab:blue', linestyle='dashdot', label='predict R')    
    # ax[4].plot(time_day.iloc[0:pos], pred[0,:pos,2], c='tab:green', label='R')
    # ax[4].plot(time_day.iloc[pos:length], pred[0,pos:,2], c='r', label='predict R')    
    ax[4].legend()
    ax[4].axvline(x=time_day[idx_end], color='k', linestyle='dashed', label='axvline')
    extraticks = [time_day[idx_end]]
    ax[4].set_xticks(list(ax[4].get_xticks())[1:-1] + extraticks)
    # plt.setp(ax[4].get_xticklabels(), rotation=45)
    ax[4].set_title(f"(e)")
    
    # ax[5].plot(time_day.iloc[0:pos], data_pred['beta'][:pos,:], c='tab:green', label='$R_0(t)$')
    # ax[5].plot(time_day.iloc[pos:length], data_pred['beta'][pos:length,:], c='r', linestyle='dashed', marker='o', markersize=1, label='predict $R_0(t)$')
    ax[5].plot(time_day, data_pred['beta'], c='tab:blue', linestyle='dashed', marker='o', markersize=1, label='predict $R_0(t)$')
    ax[5].legend()
    ax[5].axvline(x=time_day[idx_end], color='k', linestyle='dashed', label='axvline')
    extraticks = [time_day[idx_end]]
    ax[5].set_xticks(list(ax[5].get_xticks())[1:-1] + extraticks)
    # plt.setp(ax[5].get_xticklabels(), rotation=45)
    ax[5].set_title(f"(f)")
    
    fig.suptitle("Synthetic datasets", fontsize=30)

else:
    fig, ax = plt.subplots(1,5,figsize=(33,5))

    if estimate:
        ax[0].plot(time_day, data_train/N, c='r', label='Estimated I')
    else:
        ax[0].plot(time_day, data_train/N, c='r', label='Daily average I')
    ax[0].scatter(time_day.iloc[pred_idx], prediction_I/N, s=20, c='tab:blue', label=f'{pred_length} days predict I')
    ax[0].scatter(time_day.iloc[pred_idx], prediction_I_/N, s=20, facecolors='none', edgecolors='tab:green', label=f'{pred_length_} days predict I')
    ax[0].legend()
    plt.setp(ax[0].get_xticklabels(), rotation=45)
    ax[0].set_title(f"(a)")
    
    if estimate:
        me = {'Mexico':102.4, 'South Africa':106.1, 'Republic of Korea':87.4}
        # me = {'Mexico':107.8, 'South Africa':108.6, 'Republic of Korea':18}
    else:
        me = {'Mexico':60, 'South Africa':54, 'Republic of Korea':81}
    ax[1].plot(time_day.iloc[pred_idx], mu_list, linestyle='dashed', marker='o', label='$\mu$')
    n = 3 ## how many sigmas
    ax[1].fill_between(time_day.iloc[pred_idx], np.clip(mu_list-sigma_list*n,0,1000), mu_list+sigma_list*n,
        alpha=0.2, facecolor='#089FFF', #edgecolor='#1B2ACC', linewidth=1, 
        linestyle='dashdot', antialiased=True, label='3$\sigma$')
    ax[1].legend()
    ax[1].axhline(y=me[f'{country}'], color='k', linestyle='dashed', label='axvline')
    extraticks = [me[f'{country}']]
    # ax[1].set_yticks(list(ax[1].get_yticks())[3:-1] + extraticks)
    ax[1].set_yticks([200,400,600] + extraticks)
    plt.setp(ax[1].get_xticklabels(), rotation=45)
    ax[1].set_title(f"(b)")

    if estimate:
        llll = {'Mexico':60, 'South Africa':54, 'Republic of Korea':16}
    else:
        llll = {'Mexico':60, 'South Africa':2, 'Republic of Korea':19}
    l = llll[country]
    pp = path[l]
    idx_end = int(pp.split('/')[-2].split('_')[-1])
    pos = idx_end-start
    data = np.load(pp)
    pred = data['pred']/N
    
    ax[2].plot(time_day.iloc[:pos], data_train[:pos]/N, c='tab:orange', label='train I')
    ax[2].plot(time_day.iloc[pos:], data_train[pos:]/N, c='r', label='test I')
    ax[2].plot(time_day, pred[0,:,1], c='tab:blue', linestyle='dashed', label='predict I')
    ax[2].legend()
    ax[2].axvline(x=time_day[idx_end], color='k', linestyle='dashed', label='axvline')
    plt.setp(ax[2].get_xticklabels(), rotation=45)
    ax[2].set_title(f"(c)")
    
    ax[3].plot(time_day, pred[0,:,0], c='tab:blue', linestyle='dashdot', label='predict S')
    ax[3].plot(time_day, pred[0,:,2], c='tab:blue', label='predict R')
    ax[3].legend()
    ax[3].axvline(x=time_day[idx_end], color='k', linestyle='dashed', label='axvline')
    plt.setp(ax[3].get_xticklabels(), rotation=45)
    ax[3].set_title(f"(d)")

    ax[4].plot(time_day, data['beta'], c='tab:blue', linestyle='dashed', marker='o', markersize=1, label='predict $R_0(t)$')
    ax[4].legend()
    ax[4].axvline(x=time_day[idx_end], color='k', linestyle='dashed', label='axvline')
    plt.setp(ax[4].get_xticklabels(), rotation=45)
    ax[4].set_title(f"(e)")
    
    if estimate==True:
        fig.suptitle(f"{country} datasets(estimated)", fontsize=30)
    else:
        fig.suptitle(f"{country} datasets(average daily cases)", fontsize=30)

os.makedirs(f'./figures/{file_name}_prediction', exist_ok=True)
fig.tight_layout()
fig.savefig(f'./figures/{file_name}_prediction/{file_name}_{pred_length}days_prediction.png', \
            bbox_inches='tight', dpi=300)
