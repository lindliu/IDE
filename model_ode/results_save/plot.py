#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:53:17 2024

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
        'size'   : 8}
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


def load_result_path(country, start, estimate, prop, length, case='simulation_2_7'):
    ### load data
    if country!='simulation':
        data = pd.read_csv(f'../../data/covid_{country}.csv', sep='\t')
        data['date'] = pd.to_datetime(data['date'])
        
        data_train = get_train_data(data, start, length=length, recovery_time=14, estimate=estimate, prop=prop)
        data_train = data_train.flatten()
    elif country=='simulation': 
        data = pd.DataFrame(np.load(f'../../data/{case}.npy'), columns=['S','I','R'])       
        data['date'] = np.arange(data.shape[0])
         
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
                
        
    path_u = glob.glob(f'{case}/figures_end_/{file_name}_{start}_*')
    
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
    return path, data, data_train, time_day, N, file_name
    




country = 'simulation'
length = 400
start = 0
estimate, prop = True, True


fontsize=12
fig, axes = plt.subplots(4,4,figsize=(15,13))
# axs = ax.flatten()
# ax[3].axis('off')



################################# first row ##################################
ax = axes[1,:]

path, data, _, time_day, N, file_name = \
    load_result_path(country, start, estimate, prop, length, case='simulation_2_5')
data_train = data['I']

weeks = 4
pred_length = 7*weeks
pred_1, pred_2 = 2, 7
peaks = [48, 164, 270] ### three peaks of the synthetic data

point_show = np.arange(8,350,6)

pred_idx, prediction_I, prediction_I_ = [], [], []
prediction_S, prediction_R = [], []
mu_list, sigma_list, tau_list = [], [], []

# t_end = 25
# T = np.linspace(0., t_end, 400)[:length][::-1]
peak_st_sim_idx, peak_nd_sim_idx, peak_rd_sim_idx = [], [], []
model_st_date, model_nd_date, model_rd_date = [], [], []
for pp in path:
    
    idx_end = int(pp.split('/')[-2].split('_')[-1])
    pos = idx_end-start

    data_pred = np.load(pp)
    pred = data_pred['pred'][:,:length,:]
    
    if pos in point_show:
        mu_list.append(data_pred['mu'].item())
        sigma_list.append(data_pred['sigma'].item())
        tau_list.append(data_pred['tau'].item())
        
        pred_idx.append(list(np.arange(pos, pos+pred_1))[-1])
        prediction_I.append(list(pred[0,pos:pos+pred_1,1])[-1])    
        prediction_I_.append(list(pred[0,pos:pos+pred_2,1])[-1])       
        prediction_S.append(list(pred[0,pos:pos+pred_length,0])[-1])
        prediction_R.append(list(pred[0,pos:pos+pred_length,2])[-1])
    
    eps = 2  ### to control from where to select peak point
    if pos+pred_length*eps<400 and pos<347:
        ax[2].scatter(time_day.iloc[np.arange(pos,pos+pred_length*eps)], \
                      pred[0,pos:pos+pred_length*eps,1]/N, c='tab:blue',alpha=.1,s=7)
    else:
        ax[2].scatter(time_day.iloc[np.arange(pos, 400)], \
                      pred[0,pos:400,1]/N, c='tab:blue',alpha=.1,s=7)


    # if pos<peak_st[idx]-pred_length:
    if pos<peaks[0]-6 and pos>=peaks[0]-6-pred_length:
        peak_idx = np.argmax(pred[0,pos:pos+pred_length*eps,1]/N)
        ax[2].scatter(time_day.iloc[np.arange(pos,pos+pred_length*eps)[peak_idx]], \
                    pred[0,pos+peak_idx,1]/N, c='b',alpha=.9,s=10)
    
        peak_st_sim_idx.append(peak_idx+pos)
        model_st_date.append(pos)
        
    if pos<peaks[1]-6 and pos>=peaks[1]-6-pred_length:
        peak_idx = np.argmax(pred[0,pos:pos+pred_length,1]/N)
        ax[2].scatter(time_day.iloc[np.arange(pos,pos+pred_length)[peak_idx]], \
                    pred[0,pos+peak_idx,1]/N, c='b',alpha=.9,s=10)
    
        peak_nd_sim_idx.append(peak_idx+pos)
        model_nd_date.append(pos)

    if pos<peaks[2]-6 and pos>=peaks[2]-6-pred_length:
        peak_idx = np.argmax(pred[0,pos:pos+pred_length,1]/N)
        ax[2].scatter(time_day.iloc[np.arange(pos,pos+pred_length)[peak_idx]], \
                    pred[0,pos+peak_idx,1]/N, c='b',alpha=.9,s=10)
    
        peak_rd_sim_idx.append(peak_idx+pos)
        model_rd_date.append(pos)

mu_list = np.array(mu_list)
sigma_list = np.array(sigma_list)

prediction_I = np.array(prediction_I)
prediction_I_ = np.array(prediction_I_)
prediction_S = np.array(prediction_S)
prediction_R = np.array(prediction_R)

endind = int(pp.split('/')[-2].split('_')[-1])    


ax[0].plot(time_day, data_train/N, c='r', label='Synthetic j')
ax[0].scatter(time_day.iloc[pred_idx], prediction_I/N, s=10, c='tab:blue', label=f'{pred_1} days prediction')
ax[0].scatter(time_day.iloc[pred_idx], prediction_I_/N, s=10, facecolors='none', edgecolors='tab:green', label=f'{pred_2} days prediction')
ax[0].legend(fontsize=fontsize)
ax[0].set_ylim(0, ax[0].axes.get_ylim()[1]*1.5)
plt.setp(ax[0].get_xticklabels(), rotation=45)
# ax[0].set_xlabel(f"(a)",fontsize=fontsize)


ax[1].plot(time_day.iloc[pred_idx], mu_list, linestyle='dashed', marker='o', markersize=4, label='$\mu$')
n = 3 ## how many sigmas
ax[1].fill_between(time_day.iloc[pred_idx], np.clip(mu_list-sigma_list*n,0,1000), mu_list+sigma_list*n,
    alpha=0.2, facecolor='#089FFF', #edgecolor='#1B2ACC', linewidth=1, 
    linestyle='dashdot', antialiased=True, label='3$\sigma$')
ax[1].legend(fontsize=fontsize)
mu_real = 70
ax[1].axhline(y=mu_real, color='r', linestyle='dashed', label='axvline')
ax[1].set_yticks([200,400,600] + [mu_real])
ax[1].set_xlim([0,400])
ax[1].set_ylim([0,700])
# ax[1].set_ylabel('$\mu$')
plt.setp(ax[1].get_xticklabels(), rotation=45)
# ax[1].set_xlabel(f"(b)",fontsize=fontsize)


ax[2].plot(time_day, data_train/N, c='r', label='Synthetic j', markersize=10)
ax[2].legend(fontsize=fontsize)
ax[2].set_ylim(0, .4)
ax[2].axvline(x=time_day[peaks[0]+start], color='r', linestyle='dashed', label='axvline')
ax[2].axvline(x=time_day[peaks[1]+start], color='r', linestyle='dashed', label='axvline')
ax[2].axvline(x=time_day[peaks[2]+start], color='r', linestyle='dashed', label='axvline')
# ax[2].axvline(x=time_day[347], color='k', linestyle='dashed', label='axvline')
plt.setp(ax[2].get_xticklabels(), rotation=45)
# ax[2].set_xlabel(f"(c)",fontsize=fontsize)


ax[3].scatter(time_day.iloc[peak_st_sim_idx], time_day.iloc[model_st_date], c='b', s=8)
ax[3].scatter(time_day.iloc[peak_nd_sim_idx], time_day.iloc[model_nd_date], c='b', s=8)
ax[3].scatter(time_day.iloc[peak_rd_sim_idx], time_day.iloc[model_rd_date], c='b', s=8)
ax[3].set_xlim(time_day.iloc[0], time_day.iloc[399])
ax[3].set_ylim(time_day.iloc[0], time_day.iloc[330])
ax[3].axvline(x=time_day[peaks[0]+start], color='r', linestyle='dashed', label='axvline')
ax[3].axvline(x=time_day[peaks[1]+start], color='r', linestyle='dashed', label='axvline')
ax[3].axvline(x=time_day[peaks[2]+start], color='r', linestyle='dashed', label='axvline')
ax[3].axhline(y=time_day[peaks[0]+start], color='r', linestyle='dashed', label='axvline')
ax[3].axhline(y=time_day[peaks[1]+start], color='r', linestyle='dashed', label='axvline')
ax[3].axhline(y=time_day[peaks[2]+start], color='r', linestyle='dashed', label='axvline')
ax[3].axhspan(ymin=time_day[np.clip(peaks[0]+start-6-pred_length,0,400)], ymax=time_day[peaks[0]-6+start], alpha=0.3, color='gray')
ax[3].axhspan(ymin=time_day[np.clip(peaks[1]+start-6-pred_length,0,400)], ymax=time_day[peaks[1]-6+start], alpha=0.3, color='gray')
ax[3].axhspan(ymin=time_day[np.clip(peaks[2]+start-6-pred_length,0,400)], ymax=time_day[peaks[2]-6+start], alpha=0.3, color='gray')
# ax[5].set_ylabel('Model date')
plt.setp(ax[3].get_xticklabels(), rotation=45)
# ax[3].set_xlabel(f"(d)",fontsize=fontsize)








################################# second row ##################################
ax = axes[2,:]

path, data, _, time_day, N, file_name = \
    load_result_path(country, start, estimate, prop, length, case='simulation_2_7')
data_train = data['I']

weeks = 4
pred_length = 7*weeks
pred_1, pred_2 = 2, 7
peaks = [38, 167, 290] ### three peaks of the synthetic data

point_show = np.arange(8,350,6)

pred_idx, prediction_I, prediction_I_ = [], [], []
prediction_S, prediction_R = [], []
mu_list, sigma_list, tau_list = [], [], []

# t_end = 25
# T = np.linspace(0., t_end, 400)[:length][::-1]
peak_st_sim_idx, peak_nd_sim_idx, peak_rd_sim_idx = [], [], []
model_st_date, model_nd_date, model_rd_date = [], [], []
for pp in path:
    
    idx_end = int(pp.split('/')[-2].split('_')[-1])
    pos = idx_end-start

    data_pred = np.load(pp)
    pred = data_pred['pred'][:,:length,:]
    
    if pos in point_show:
        mu_list.append(data_pred['mu'].item())
        sigma_list.append(data_pred['sigma'].item())
        tau_list.append(data_pred['tau'].item())
        
        pred_idx.append(list(np.arange(pos, pos+pred_1))[-1])
        prediction_I.append(list(pred[0,pos:pos+pred_1,1])[-1])    
        prediction_I_.append(list(pred[0,pos:pos+pred_2,1])[-1])       
        prediction_S.append(list(pred[0,pos:pos+pred_length,0])[-1])
        prediction_R.append(list(pred[0,pos:pos+pred_length,2])[-1])
    
    eps = 2  ### to control from where to select peak point
    if pos+pred_length*eps<400 and pos<347:
        ax[2].scatter(time_day.iloc[np.arange(pos,pos+pred_length*eps)], \
                      pred[0,pos:pos+pred_length*eps,1]/N, c='tab:blue',alpha=.1,s=7)
    else:
        ax[2].scatter(time_day.iloc[np.arange(pos, 400)], \
                      pred[0,pos:400,1]/N, c='tab:blue',alpha=.1,s=7)


    # if pos<peak_st[idx]-pred_length:
    if pos<peaks[0]-6 and pos>=peaks[0]-6-pred_length:
        peak_idx = np.argmax(pred[0,pos:pos+pred_length*eps,1]/N)
        ax[2].scatter(time_day.iloc[np.arange(pos,pos+pred_length*eps)[peak_idx]], \
                    pred[0,pos+peak_idx,1]/N, c='b',alpha=.9,s=10)
    
        peak_st_sim_idx.append(peak_idx+pos)
        model_st_date.append(pos)
        
    if pos<peaks[1]-6 and pos>=peaks[1]-6-pred_length:
        peak_idx = np.argmax(pred[0,pos:pos+pred_length,1]/N)
        ax[2].scatter(time_day.iloc[np.arange(pos,pos+pred_length)[peak_idx]], \
                    pred[0,pos+peak_idx,1]/N, c='b',alpha=.9,s=10)
    
        peak_nd_sim_idx.append(peak_idx+pos)
        model_nd_date.append(pos)

    if pos<peaks[2]-6 and pos>=peaks[2]-6-pred_length:
        peak_idx = np.argmax(pred[0,pos:pos+pred_length,1]/N)
        ax[2].scatter(time_day.iloc[np.arange(pos,pos+pred_length)[peak_idx]], \
                    pred[0,pos+peak_idx,1]/N, c='b',alpha=.9,s=10)
    
        peak_rd_sim_idx.append(peak_idx+pos)
        model_rd_date.append(pos)

mu_list = np.array(mu_list)
sigma_list = np.array(sigma_list)

prediction_I = np.array(prediction_I)
prediction_I_ = np.array(prediction_I_)
prediction_S = np.array(prediction_S)
prediction_R = np.array(prediction_R)

endind = int(pp.split('/')[-2].split('_')[-1])    


ax[0].plot(time_day, data_train/N, c='r', label='Synthetic j')
ax[0].scatter(time_day.iloc[pred_idx], prediction_I/N, s=10, c='tab:blue', label=f'{pred_1} days prediction')
ax[0].scatter(time_day.iloc[pred_idx], prediction_I_/N, s=10, facecolors='none', edgecolors='tab:green', label=f'{pred_2} days prediction')
ax[0].legend(fontsize=fontsize)
ax[0].set_ylim(0, ax[0].axes.get_ylim()[1]*1.5)
plt.setp(ax[0].get_xticklabels(), rotation=45)
# ax[0].set_xlabel(f"(a)",fontsize=fontsize)


ax[1].plot(time_day.iloc[pred_idx], mu_list, linestyle='dashed', marker='o', markersize=4, label='$\mu$')
n = 3 ## how many sigmas
ax[1].fill_between(time_day.iloc[pred_idx], np.clip(mu_list-sigma_list*n,0,1000), mu_list+sigma_list*n,
    alpha=0.2, facecolor='#089FFF', #edgecolor='#1B2ACC', linewidth=1, 
    linestyle='dashdot', antialiased=True, label='3$\sigma$')
ax[1].legend(fontsize=fontsize)
mu_real = 70
ax[1].axhline(y=mu_real, color='r', linestyle='dashed', label='axvline')
ax[1].set_yticks([200,400,600] + [mu_real])
ax[1].set_xlim([0,400])
ax[1].set_ylim([0,700])
# ax[1].set_ylabel('$\mu$')
plt.setp(ax[1].get_xticklabels(), rotation=45)
# ax[1].set_xlabel(f"(b)",fontsize=fontsize)


ax[2].plot(time_day, data_train/N, c='r', label='Synthetic j', markersize=10)
ax[2].legend(fontsize=fontsize)
ax[2].set_ylim(0, .4)
ax[2].axvline(x=time_day[peaks[0]+start], color='r', linestyle='dashed', label='axvline')
ax[2].axvline(x=time_day[peaks[1]+start], color='r', linestyle='dashed', label='axvline')
ax[2].axvline(x=time_day[peaks[2]+start], color='r', linestyle='dashed', label='axvline')
# ax[2].axvline(x=time_day[347], color='k', linestyle='dashed', label='axvline')
plt.setp(ax[2].get_xticklabels(), rotation=45)
# ax[2].set_xlabel(f"(c)",fontsize=fontsize)


ax[3].scatter(time_day.iloc[peak_st_sim_idx], time_day.iloc[model_st_date], c='b', s=8)
ax[3].scatter(time_day.iloc[peak_nd_sim_idx], time_day.iloc[model_nd_date], c='b', s=8)
ax[3].scatter(time_day.iloc[peak_rd_sim_idx], time_day.iloc[model_rd_date], c='b', s=8)
ax[3].set_xlim(time_day.iloc[0], time_day.iloc[399])
ax[3].set_ylim(time_day.iloc[0], time_day.iloc[330])
ax[3].axvline(x=time_day[peaks[0]+start], color='r', linestyle='dashed', label='axvline')
ax[3].axvline(x=time_day[peaks[1]+start], color='r', linestyle='dashed', label='axvline')
ax[3].axvline(x=time_day[peaks[2]+start], color='r', linestyle='dashed', label='axvline')
ax[3].axhline(y=time_day[peaks[0]+start], color='r', linestyle='dashed', label='axvline')
ax[3].axhline(y=time_day[peaks[1]+start], color='r', linestyle='dashed', label='axvline')
ax[3].axhline(y=time_day[peaks[2]+start], color='r', linestyle='dashed', label='axvline')
ax[3].axhspan(ymin=time_day[np.clip(peaks[0]+start-6-pred_length,0,400)], ymax=time_day[peaks[0]-6+start], alpha=0.3, color='gray')
ax[3].axhspan(ymin=time_day[np.clip(peaks[1]+start-6-pred_length,0,400)], ymax=time_day[peaks[1]-6+start], alpha=0.3, color='gray')
ax[3].axhspan(ymin=time_day[np.clip(peaks[2]+start-6-pred_length,0,400)], ymax=time_day[peaks[2]-6+start], alpha=0.3, color='gray')
# ax[5].set_ylabel('Model date')
plt.setp(ax[3].get_xticklabels(), rotation=45)
# ax[3].set_xlabel(f"(d)",fontsize=fontsize)













################################# second row ##################################
ax = axes[3,:]

path, data, _, time_day, N, file_name = \
    load_result_path(country, start, estimate, prop, length, case='simulation_2_9')
data_train = data['I']

weeks = 4
pred_length = 7*weeks
pred_1, pred_2 = 2, 7
peaks = [48, 193, 321] ### three peaks of the synthetic data

point_show = np.arange(8,350,6)

pred_idx, prediction_I, prediction_I_ = [], [], []
prediction_S, prediction_R = [], []
mu_list, sigma_list, tau_list = [], [], []

# t_end = 25
# T = np.linspace(0., t_end, 400)[:length][::-1]
peak_st_sim_idx, peak_nd_sim_idx, peak_rd_sim_idx = [], [], []
model_st_date, model_nd_date, model_rd_date = [], [], []
for pp in path:
    
    idx_end = int(pp.split('/')[-2].split('_')[-1])
    pos = idx_end-start

    data_pred = np.load(pp)
    pred = data_pred['pred'][:,:length,:]
    
    if pos in point_show:
        mu_list.append(data_pred['mu'].item())
        sigma_list.append(data_pred['sigma'].item())
        tau_list.append(data_pred['tau'].item())
        
        pred_idx.append(list(np.arange(pos, pos+pred_1))[-1])
        prediction_I.append(list(pred[0,pos:pos+pred_1,1])[-1])    
        prediction_I_.append(list(pred[0,pos:pos+pred_2,1])[-1])       
        prediction_S.append(list(pred[0,pos:pos+pred_length,0])[-1])
        prediction_R.append(list(pred[0,pos:pos+pred_length,2])[-1])
    
    eps = 2  ### to control from where to select peak point
    if pos+pred_length*eps<400 and pos<347:
        ax[2].scatter(time_day.iloc[np.arange(pos,pos+pred_length*eps)], \
                      pred[0,pos:pos+pred_length*eps,1]/N, c='tab:blue',alpha=.1,s=7)
    else:
        ax[2].scatter(time_day.iloc[np.arange(pos, 400)], \
                      pred[0,pos:400,1]/N, c='tab:blue',alpha=.1,s=7)


    # if pos<peak_st[idx]-pred_length:
    if pos<peaks[0]-6 and pos>=peaks[0]-6-pred_length:
        peak_idx = np.argmax(pred[0,pos:pos+pred_length*eps,1]/N)
        ax[2].scatter(time_day.iloc[np.arange(pos,pos+pred_length*eps)[peak_idx]], \
                    pred[0,pos+peak_idx,1]/N, c='b',alpha=.9,s=10)
    
        peak_st_sim_idx.append(peak_idx+pos)
        model_st_date.append(pos)
        
    if pos<peaks[1]-6 and pos>=peaks[1]-6-pred_length:
        peak_idx = np.argmax(pred[0,pos:pos+pred_length,1]/N)
        ax[2].scatter(time_day.iloc[np.arange(pos,pos+pred_length)[peak_idx]], \
                    pred[0,pos+peak_idx,1]/N, c='b',alpha=.9,s=10)
    
        peak_nd_sim_idx.append(peak_idx+pos)
        model_nd_date.append(pos)

    if pos<peaks[2]-6 and pos>=peaks[2]-6-pred_length:
        peak_idx = np.argmax(pred[0,pos:pos+pred_length,1]/N)
        ax[2].scatter(time_day.iloc[np.arange(pos,pos+pred_length)[peak_idx]], \
                    pred[0,pos+peak_idx,1]/N, c='b',alpha=.9,s=10)
    
        peak_rd_sim_idx.append(peak_idx+pos)
        model_rd_date.append(pos)

mu_list = np.array(mu_list)
sigma_list = np.array(sigma_list)

prediction_I = np.array(prediction_I)
prediction_I_ = np.array(prediction_I_)
prediction_S = np.array(prediction_S)
prediction_R = np.array(prediction_R)

endind = int(pp.split('/')[-2].split('_')[-1])    


ax[0].plot(time_day, data_train/N, c='r', label='Synthetic j')
ax[0].scatter(time_day.iloc[pred_idx], prediction_I/N, s=10, c='tab:blue', label=f'{pred_1} days prediction')
ax[0].scatter(time_day.iloc[pred_idx], prediction_I_/N, s=10, facecolors='none', edgecolors='tab:green', label=f'{pred_2} days prediction')
ax[0].legend(fontsize=fontsize)
ax[0].set_ylim(0, ax[0].axes.get_ylim()[1]*1.5)
plt.setp(ax[0].get_xticklabels(), rotation=45)
# ax[0].set_xlabel(f"(a)",fontsize=fontsize)


ax[1].plot(time_day.iloc[pred_idx], mu_list, linestyle='dashed', marker='o', markersize=4, label='$\mu$')
n = 3 ## how many sigmas
ax[1].fill_between(time_day.iloc[pred_idx], np.clip(mu_list-sigma_list*n,0,1000), mu_list+sigma_list*n,
    alpha=0.2, facecolor='#089FFF', #edgecolor='#1B2ACC', linewidth=1, 
    linestyle='dashdot', antialiased=True, label='3$\sigma$')
ax[1].legend(fontsize=fontsize)
mu_real = 80
ax[1].axhline(y=mu_real, color='r', linestyle='dashed', label='axvline')
ax[1].set_yticks([200,400,600] + [mu_real])
ax[1].set_xlim([0,400])
ax[1].set_ylim([0,700])
# ax[1].set_ylabel('$\mu$')
plt.setp(ax[1].get_xticklabels(), rotation=45)
# ax[1].set_xlabel(f"(b)",fontsize=fontsize)


ax[2].plot(time_day, data_train/N, c='r', label='Synthetic j', markersize=10)
ax[2].legend(fontsize=fontsize)
ax[2].set_ylim(0, .4)
ax[2].axvline(x=time_day[peaks[0]+start], color='r', linestyle='dashed', label='axvline')
ax[2].axvline(x=time_day[peaks[1]+start], color='r', linestyle='dashed', label='axvline')
ax[2].axvline(x=time_day[peaks[2]+start], color='r', linestyle='dashed', label='axvline')
# ax[2].axvline(x=time_day[347], color='k', linestyle='dashed', label='axvline')
plt.setp(ax[2].get_xticklabels(), rotation=45)
# ax[2].set_xlabel(f"(c)",fontsize=fontsize)


ax[3].scatter(time_day.iloc[peak_st_sim_idx], time_day.iloc[model_st_date], c='b', s=8)
ax[3].scatter(time_day.iloc[peak_nd_sim_idx], time_day.iloc[model_nd_date], c='b', s=8)
ax[3].scatter(time_day.iloc[peak_rd_sim_idx], time_day.iloc[model_rd_date], c='b', s=8)
ax[3].set_xlim(time_day.iloc[0], time_day.iloc[399])
ax[3].set_ylim(time_day.iloc[0], time_day.iloc[330])
ax[3].axvline(x=time_day[peaks[0]+start], color='r', linestyle='dashed', label='axvline')
ax[3].axvline(x=time_day[peaks[1]+start], color='r', linestyle='dashed', label='axvline')
ax[3].axvline(x=time_day[peaks[2]+start], color='r', linestyle='dashed', label='axvline')
ax[3].axhline(y=time_day[peaks[0]+start], color='r', linestyle='dashed', label='axvline')
ax[3].axhline(y=time_day[peaks[1]+start], color='r', linestyle='dashed', label='axvline')
ax[3].axhline(y=time_day[peaks[2]+start], color='r', linestyle='dashed', label='axvline')
ax[3].axhspan(ymin=time_day[np.clip(peaks[0]+start-6-pred_length,0,400)], ymax=time_day[peaks[0]-6+start], alpha=0.3, color='gray')
ax[3].axhspan(ymin=time_day[np.clip(peaks[1]+start-6-pred_length,0,400)], ymax=time_day[peaks[1]-6+start], alpha=0.3, color='gray')
ax[3].axhspan(ymin=time_day[np.clip(peaks[2]+start-6-pred_length,0,400)], ymax=time_day[peaks[2]-6+start], alpha=0.3, color='gray')
# ax[5].set_ylabel('Model date')
plt.setp(ax[3].get_xticklabels(), rotation=45)
# ax[3].set_xlabel(f"(d)",fontsize=fontsize)




# fig.tight_layout()
# fig.savefig(f'./figures/prediction_synthetic.png', bbox_inches='tight', dpi=300)

    


# print('first peak:')
# error_S_st = np.median(abs(np.array(peak_st_sim_idx)-peaks[0]).reshape(weeks,7), axis=1)
# print(f'Simulation: {error_S_st}')

# print('Second peak:')
# error_S_nd = np.median(abs(np.array(peak_nd_sim_idx)-peaks[1]).reshape(weeks,7), axis=1)
# print(f'Simulation: {error_S_nd}')

# # print('Third peak:')
# # error_S_rd = np.median(abs(np.array(peak_rd_sim_idx)-peaks[2]).reshape(weeks,7), axis=1)
# # print(f'Simulation: {error_S_rd}')

