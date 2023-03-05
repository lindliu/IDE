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


i = -1
estimate = True

start_list = [750, 655, 750, 630, 710, 0]
countries = ['United Kingdom', 'Mexico', 'Belgium', \
             'South Africa', 'Republic of Korea',\
             'simulation']
country = countries[i]

length = 400
### load data
if country!='simulation':
    data_train_ = pd.read_csv(f'../data/covid_{country}.csv', sep='\t')
    data_train_['date'] = pd.to_datetime(data_train_['date'])

    start = start_list[i]
    data_train = data_train_['proportion'][start:start+length].to_numpy()

elif country=='simulation': 
    data_train_ = pd.DataFrame(np.load('../data/simulation_2_3.npy'), columns=['S','I','R'])        
    # data_train_["date"] = pd.date_range(start='1/1/2021', periods=500)   
    data_train_["date"] = np.arange(500)

    start = start_list[i]
    data_train = data_train_['I'][start:start+length].to_numpy()
# data_train_ = pd.read_csv(f'../data/covid_{country}.csv', sep='\t')
# data_train_['date'] = pd.to_datetime(data_train_['date'])


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
    
    
data = np.load(path[0])
print(list(data.keys()))



fig, ax = plt.subplots(1,4,figsize=(16,4))

time_day = data_train_['date'][start:start+length]

pred_length = 7
pred_idx, prediction_I = [], []
prediction_S, prediction_R = [], []
mu_list, sigma_list = [], []

t_end = 25
T = np.linspace(0., t_end, length)[::-1]
                   
for pp in path:
    
    idx_end = int(pp.split('/')[-2].split('_')[-1])
    data = np.load(pp)
    pred = data['pred']
    
    mu_list.append(data['mu'].item())
    sigma_list.append(data['sigma'].item())
    # pred_idx.extend(list(np.arange(idx_end-start, idx_end-start+pred_length)))
    # prediction_I.extend(list(pred[0,idx_end-start:idx_end-start+pred_length,1]))
    
    pred_idx.append(list(np.arange(idx_end-start, idx_end-start+pred_length))[-1])
    prediction_I.append(list(pred[0,idx_end-start:idx_end-start+pred_length,1])[-1])
    
    prediction_S.append(list(pred[0,idx_end-start:idx_end-start+pred_length,0])[-1])
    prediction_R.append(list(pred[0,idx_end-start:idx_end-start+pred_length,2])[-1])
    
    # ax[1].scatter(time_day.iloc[np.arange(idx_end-start, idx_end-start+pred_length)], \
    #             pred[0,idx_end-start:idx_end-start+pred_length,1], s=1)

mu_list = np.array(mu_list)
sigma_list = np.array(sigma_list)

# ax[0].scatter(time_day.iloc[pred_idx], prediction_S, s=1, c='r', label='estimated suseptible')
ax[0].plot(time_day, pred[0,:,0], label='predicted suseptible')
ax[0].legend()
plt.setp(ax[0].get_xticklabels(), rotation=45)
# ax[0].set_title(f"{country}")

    
ax[1].plot(time_day, data_train, label='estimated infectious')
ax[1].scatter(time_day.iloc[pred_idx], prediction_I, s=1, c='r', label=f'{pred_length} days prediction')
ax[1].legend()
plt.setp(ax[1].get_xticklabels(), rotation=45)
# ax[1].set_title(f"{country}")

# ax[2].scatter(time_day.iloc[pred_idx], prediction_R, s=1, c='r', label='estimated recovered')
ax[2].plot(time_day, pred[0,:,2], label='predicted recovered')
ax[2].legend()
plt.setp(ax[2].get_xticklabels(), rotation=45)
# ax[2].set_title(f"{country}")


scale = 400/t_end
# ax[3].scatter(time_day.iloc[pred_idx], np.array(mu_list)*scale, s=1, c='r', label='$\mu$')
ax[3].plot(time_day.iloc[pred_idx], mu_list*scale, linestyle='dashed', marker='o', label='$\mu$')
ax[3].legend()
# ax[3].set_title(f"{country}")

n = 3 ## how many sigmas
ax[3].fill_between(time_day.iloc[pred_idx], (mu_list-sigma_list*n)*scale, (mu_list+sigma_list*n)*scale,
    alpha=0.2, facecolor='#089FFF', #edgecolor='#1B2ACC', linewidth=1, 
    linestyle='dashdot', antialiased=True)

plt.setp(ax[3].get_xticklabels(), rotation=45)

fig.suptitle(f"{country} datasets")
    
os.makedirs(f'./figures/{file_name}_prediction', exist_ok=True)
fig.savefig(f'./figures/{file_name}_prediction/{country}_{pred_length}days_prediction.png', bbox_inches='tight', dpi=600)









