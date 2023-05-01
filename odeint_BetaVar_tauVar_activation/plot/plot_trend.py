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
        data = pd.read_csv(f'../../data/covid_{country}.csv', sep='\t')
        data['date'] = pd.to_datetime(data['date'])
        
        data_train = get_train_data(data, start, length=length, recovery_time=14, estimate=estimate, prop=prop)
        data_train = data_train.flatten()
    elif country=='simulation': 
        data = pd.DataFrame(np.load('../../data/simulation_2_3.npy'), columns=['S','I','R'])       
        data['date'] = np.arange(500)
         
        data = data.iloc[start:start+length,:]
        data_train = data['I']
    
    time_day = data['date'][start:start+length]    
    
    if prop or country=='simulation':
        N = 1
    else:
        N =  int(data['population'].iloc[0])

    return data_train, time_day, N


countries = ['Mexico', 'South Africa', 'Republic of Korea', \
             'Mexico', 'South Africa', 'Republic of Korea']  #'simulation'
start_list = [640, 640, 640, 640, 640, 640]
length = 400

fig, ax = plt.subplots(6,3, figsize=[30,35])
ax = ax.flatten()
for idx in range(6):
        country = countries[idx]
        start = start_list[idx]
        
        estimate, prop = (False, False) if idx<=3 else (True, True)
        if country=='simulation':
            filename = f'{country}_{start}'
        else:
            filename = f'estimate_{country}_{start}' if estimate else f'real_{country}_{start}'
                  
        data_train, time_day, N = load_train_data(country, start, estimate, prop, length)
        data_train = data_train/N

        path_peak_1 = glob.glob(f'../figures_trend/{filename}_*_first_peak/*.npz')[0]
        data_peak_1 = np.load(path_peak_1)
        pred_peak_1 = data_peak_1['pred'][0]/N
        beta_peak_1 = data_peak_1['beta']

        endind_peak_1 = int(path_peak_1.split('/')[-2].split('_')[-3])
        pos_peak_1 = endind_peak_1-start

        path_peak_2 = glob.glob(f'../figures_trend/{filename}_*_second_peak/*.npz')[0]
        data_peak_2 = np.load(path_peak_2)
        pred_peak_2 = data_peak_2['pred'][0]/N
        beta_peak_2 = data_peak_2['beta']

        endind_peak_2 = int(path_peak_2.split('/')[-2].split('_')[-3])
        pos_peak_2 = endind_peak_2-start

        # idx = 0
        ax[idx*3].plot(time_day, data_train, c='r', label='estimated I')
        ax[idx*3].plot(time_day.iloc[:pos_peak_2], pred_peak_1[:pos_peak_2,1], c='b', linestyle='dashdot', label='predict I (1st peak)')
        ax[idx*3].plot(time_day, pred_peak_2[:,1], c='darkgreen', linestyle='dotted', label='predict I (2nd peak)')

        ax[idx*3].legend()
        ax[idx*3].axvline(x=time_day[endind_peak_1], color='k', linestyle='dashed', label='axvline')
        ax[idx*3].axvline(x=time_day[endind_peak_2], color='k', linestyle='dashed', label='axvline')

        ax[idx*3].axvspan(time_day.iloc[pos_peak_1], time_day.iloc[pos_peak_2], alpha=0.15, color='b')
        ax[idx*3].axvspan(time_day.iloc[pos_peak_2], time_day.iloc[-1], alpha=0.15, color='g')
        plt.setp(ax[idx*3].get_xticklabels(), rotation=45)


        ax[idx*3+1].plot(time_day.iloc[:pos_peak_2], pred_peak_1[:pos_peak_2, 0], c='b', label='S (1st peak)')
        ax[idx*3+1].plot(time_day.iloc[:pos_peak_2], pred_peak_1[:pos_peak_2, 2], c='b', linestyle='dashed', label='R (1st peak)')

        ax[idx*3+1].plot(time_day, pred_peak_2[:, 0], c='darkgreen', label='S (2nd peak)')
        ax[idx*3+1].plot(time_day, pred_peak_2[:, 2], c='darkgreen', linestyle='dashed', label='R (2nd peak)')

        ax[idx*3+1].legend()
        ax[idx*3+1].axvline(x=time_day[endind_peak_1], color='k', linestyle='dashed', label='axvline')
        ax[idx*3+1].axvline(x=time_day[endind_peak_2], color='k', linestyle='dashed', label='axvline')

        ax[idx*3+1].axvspan(time_day.iloc[pos_peak_1], time_day.iloc[pos_peak_2], alpha=0.15, color='b')
        ax[idx*3+1].axvspan(time_day.iloc[pos_peak_2], time_day.iloc[-1], alpha=0.15, color='g')
        plt.setp(ax[idx*3+1].get_xticklabels(), rotation=45)


        ax[idx*3+2].plot(time_day.iloc[:pos_peak_2], beta_peak_1[:pos_peak_2, 0], c='b', label='$R_0$ (1st peak)')
        ax[idx*3+2].plot(time_day, beta_peak_2[:, 0], c='darkgreen', label='$R_0$ (2nd peak)')

        ax[idx*3+2].legend()
        ax[idx*3+2].axvline(x=time_day[endind_peak_1], color='k', linestyle='dashed', label='axvline')
        ax[idx*3+2].axvline(x=time_day[endind_peak_2], color='k', linestyle='dashed', label='axvline')

        ax[idx*3+2].axvspan(time_day.iloc[pos_peak_1], time_day.iloc[pos_peak_2], alpha=0.15, color='b')
        ax[idx*3+2].axvspan(time_day.iloc[pos_peak_2], time_day.iloc[-1], alpha=0.15, color='g')
        plt.setp(ax[idx*3+2].get_xticklabels(), rotation=45)

ax = ax.reshape(6,3)
pad = 5
rows = ['Mexico\n(estimated datasets)', 'South Africa\n(estimated datasets)', 'Republic of Korea\n(estimated datasets)',\
        'Mexico\n(average daily cases)', 'South Africa\n(average daily cases)', 'Republic of Korea\n(average daily cases)']
for ax_, row in zip(ax[:,0], rows):
    ax_.annotate(row, xy=(0, 0.5), xytext=(-ax_.yaxis.labelpad - pad, 0),
                xycoords=ax_.yaxis.label, textcoords='offset points',
                fontsize=30, ha='right', va='center', rotation=90)

cols = ['I prediction', 'S and R prediction', 'Estimated $R_0(t)$']
for ax_, col in zip(ax[0], cols):
    ax_.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                fontsize=30, ha='center', va='baseline')

fig.tight_layout()
fig.savefig(f'./prediction_trend.png', \
            bbox_inches='tight', dpi=300)

    
    











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
    if country=='simulation': 
        data = pd.DataFrame(np.load('../../data/simulation_2_3.npy'), columns=['S','I','R'])       
        data['date'] = np.arange(500)
         
        data = data.iloc[start:start+length,:]
    
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
                
        
    path_u = glob.glob(f'../figures_/{file_name}_{start}_*')
    
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
    return path, data, time_day, N, file_name
    
def load_results(path, pred_length=2, pred_length_=7):
    pred_idx, prediction_I, prediction_I_ = [], [], []
    prediction_S, prediction_R = [], []
    mu_list, sigma_list = [], []
    tau_list = []
    
    t_end = 25
    # T = np.linspace(0., t_end, 400)[:length][::-1]
                       
    for pp in path:
        
        idx_end = int(pp.split('/')[-2].split('_')[-1])
        pos = idx_end-start
    
        data_pred = np.load(pp)
        pred = data_pred['pred'][:,:length,:]
            
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

    return pred_idx, mu_list, sigma_list, prediction_I, prediction_I_, prediction_S, prediction_R





country = 'simulation'
length = 400
start = 0

estimate, prop = True, True

path, data, time_day, N, file_name = load_result_path(country, start, estimate, prop, length)
data_train = data['I']
pred_length, pred_length_ = 2, 7
pred_idx, mu_list, sigma_list, prediction_I, prediction_I_, prediction_S, prediction_R = load_results(path, pred_length, pred_length_)


fig, ax = plt.subplots(2,4,figsize=(30,15))
ax = ax.flatten()

ax[0].axis('off')
ax[3].axis('off')

ax[1].plot(time_day, data_train/N, c='r', label='Synthetic I')
ax[1].scatter(time_day.iloc[pred_idx], prediction_I/N, s=20, c='tab:blue', label=f'{pred_length} days predict I')
ax[1].scatter(time_day.iloc[pred_idx], prediction_I_/N, s=20, facecolors='none', edgecolors='tab:green', label=f'{pred_length_} days predict I')
ax[1].legend()
plt.setp(ax[1].get_xticklabels(), rotation=45)
ax[1].set_xlabel(f"(a)")


ax[2].plot(time_day.iloc[pred_idx], mu_list, linestyle='dashed', marker='o', label='$\mu$')
n = 3 ## how many sigmas
ax[2].fill_between(time_day.iloc[pred_idx], np.clip(mu_list-sigma_list*n,0,1000), mu_list+sigma_list*n,
    alpha=0.2, facecolor='#089FFF', #edgecolor='#1B2ACC', linewidth=1, 
    linestyle='dashdot', antialiased=True, label='3$\sigma$')
ax[2].legend()
mu_real = 70
ax[2].axhline(y=mu_real, color='r', linestyle='dashed', label='axvline')
ax[2].set_yticks([200,400,600] + [mu_real])
ax[2].set_xlim([0,400])
plt.setp(ax[2].get_xticklabels(), rotation=45)
ax[2].set_xlabel(f"(b)")


filename = f'{country}_{start}'
path_peak_1 = glob.glob(f'../figures_trend/{filename}_*_first_peak/*.npz')[0]
data_peak_1 = np.load(path_peak_1)
pred_peak_1 = data_peak_1['pred'][0]/N
beta_peak_1 = data_peak_1['beta']

endind_peak_1 = int(path_peak_1.split('/')[-2].split('_')[-3])
pos_peak_1 = endind_peak_1-start

path_peak_2 = glob.glob(f'../figures_trend/{filename}_*_second_peak/*.npz')[0]
data_peak_2 = np.load(path_peak_2)
pred_peak_2 = data_peak_2['pred'][0]/N
beta_peak_2 = data_peak_2['beta']

endind_peak_2 = int(path_peak_2.split('/')[-2].split('_')[-3])
pos_peak_2 = endind_peak_2-start

# idx = 0
ax[4].plot(time_day, data_train, c='r', label='estimated I')
ax[4].plot(time_day.iloc[:pos_peak_2], pred_peak_1[:pos_peak_2,1], c='b', linestyle='dashdot', label='predict I (1st peak)')
ax[4].plot(time_day, pred_peak_2[:,1], c='darkgreen', linestyle='dotted', label='predict I (2nd peak)')

ax[4].legend()
ax[4].axvline(x=time_day[endind_peak_1], color='k', linestyle='dashed', label='axvline')
ax[4].axvline(x=time_day[endind_peak_2], color='k', linestyle='dashed', label='axvline')

ax[4].axvspan(time_day.iloc[pos_peak_1], time_day.iloc[pos_peak_2], alpha=0.15, color='b')
ax[4].axvspan(time_day.iloc[pos_peak_2], time_day.iloc[-1], alpha=0.15, color='g')
plt.setp(ax[4].get_xticklabels(), rotation=45)
ax[4].set_xlabel(f"(c)")


ax[5].plot(time_day, data['S']/N, c='r', label='Synthetic S')
ax[5].plot(time_day.iloc[:pos_peak_2], pred_peak_1[:pos_peak_2, 0], c='b', label='S (1st peak)')
ax[5].plot(time_day, pred_peak_2[:, 0], c='darkgreen', label='S (2nd peak)')
ax[5].legend()
ax[5].axvline(x=time_day[endind_peak_1], color='k', linestyle='dashed', label='axvline')
ax[5].axvspan(time_day.iloc[pos_peak_1], time_day.iloc[pos_peak_2], alpha=0.15, color='b')
ax[5].axvline(x=time_day[endind_peak_2], color='k', linestyle='dashed', label='axvline')
ax[5].axvspan(time_day.iloc[pos_peak_2], time_day.iloc[-1], alpha=0.15, color='g')
plt.setp(ax[5].get_xticklabels(), rotation=45)
ax[5].set_xlabel(f"(d)")

ax[6].plot(time_day, data['R']/N, c='r', label='Synthetic R')
ax[6].plot(time_day.iloc[:pos_peak_2], pred_peak_1[:pos_peak_2, 2], c='b', linestyle='dashed', label='R (1st peak)')
ax[6].plot(time_day, pred_peak_2[:, 2], c='darkgreen', linestyle='dashed', label='R (2nd peak)')
ax[6].legend()
ax[6].axvline(x=time_day[endind_peak_1], color='k', linestyle='dashed', label='axvline')
ax[6].axvspan(time_day.iloc[pos_peak_1], time_day.iloc[pos_peak_2], alpha=0.15, color='b')
ax[6].axvline(x=time_day[endind_peak_2], color='k', linestyle='dashed', label='axvline')
ax[6].axvspan(time_day.iloc[pos_peak_2], time_day.iloc[-1], alpha=0.15, color='g')
plt.setp(ax[6].get_xticklabels(), rotation=45)
ax[6].set_xlabel(f"(e)")


ax[7].plot(time_day.iloc[:pos_peak_2], beta_peak_1[:pos_peak_2, 0], c='b', label='$R_0$ (1st peak)')
ax[7].plot(time_day, beta_peak_2[:, 0], c='darkgreen', label='$R_0$ (2nd peak)')

ax[7].legend()
ax[7].axvline(x=time_day[endind_peak_1], color='k', linestyle='dashed', label='axvline')
ax[7].axvline(x=time_day[endind_peak_2], color='k', linestyle='dashed', label='axvline')

ax[7].axvspan(time_day.iloc[pos_peak_1], time_day.iloc[pos_peak_2], alpha=0.15, color='b')
ax[7].axvspan(time_day.iloc[pos_peak_2], time_day.iloc[-1], alpha=0.15, color='g')

beta_real = 2.3
ax[7].axhline(y=beta_real, color='r', linestyle='dashed', label='axvline')
ax[7].set_yticks(list(np.arange(0,4,.5)) + [beta_real])
plt.setp(ax[7].get_xticklabels(), rotation=45)
ax[7].set_xlabel(f"(f)")


fig.tight_layout()
fig.savefig(f'./prediction_trend_synthetic.png', \
            bbox_inches='tight', dpi=300)

    

