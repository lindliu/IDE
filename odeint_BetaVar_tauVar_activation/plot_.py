#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 10:19:57 2023

@author: dliu
"""

##pip install mpl-scatter-density
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pandas as pd


# #### example #####
# # "Viridis-like" colormap with white background
# white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
#     (0, '#ffffff'),
#     (1e-20, '#440053'),
#     (0.2, '#404388'),
#     (0.4, '#2a788e'),
#     (0.6, '#21a784'),
#     (0.8, '#78d151'),
#     (1, '#fde624'),
# ], N=256)

# def using_mpl_scatter_density(fig, x, y):
#     ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
#     density = ax.scatter_density(x, y, cmap=white_viridis)
#     fig.colorbar(density, label='Number of points per pixel')


# # Fake data for testing
# x = np.random.normal(size=100000)
# y = x * 3 + np.random.normal(size=100000)

# fig = plt.figure()
# using_mpl_scatter_density(fig, x, y)
# plt.show()
# ##################


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



time_day = data_train_['date'][start:start+length]

pred_length = 10
idx_list, I_list = [], []
S_list, R_list = [], []
mu_list, sigma_list = [], []

t_end = 25
T = np.linspace(0., t_end, length)[::-1]
                   
for pp in path[:]:
    
    idx_end = int(pp.split('/')[-2].split('_')[-1])
    # idx_end = 400
    
    data = np.load(pp)
    pred = data['pred']
    
    mu_list.append(data['mu'].item())
    sigma_list.append(data['sigma'].item())
    
    S_list.extend(list(pred[0,idx_end-start:idx_end-start+pred_length,0]))    
    I_list.extend(list(pred[0,idx_end-start:idx_end-start+pred_length,1]))
    R_list.extend(list(pred[0,idx_end-start:idx_end-start+pred_length,2]))
    
    idx_list.extend(list(np.arange(idx_end-start, idx_end-start+pred_length)))

    # ax[1].scatter(time_day.iloc[np.arange(idx_end-start, idx_end-start+pred_length)], \
    #             pred[0,idx_end-start:idx_end-start+pred_length,1], s=1)

mu_list = np.array(mu_list)
sigma_list = np.array(sigma_list)


white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.001, '#404388'),
    (0.005, '#2a788e'),
    (0.01, '#21a784'),
    (0.1, '#78d151'),
    (0.2, '#fde624'),
    (1, '#ffffff'),
], N=256)


# fig = plt.figure()

# ax1 = fig.add_subplot(1, 3, 1, projection='scatter_density')
# ax2 = fig.add_subplot(1, 3, 2, projection='scatter_density')
# ax3 = fig.add_subplot(1, 3, 3, projection='scatter_density')
# ax = [ax1, ax2, ax3]

# # density = ax.scatter_density(x, y, cmap=white_viridis)
    
# # fig, ax = plt.subplots(1,4,figsize=(16,4))

# # ax[0].plot(time_day[idx_list], pred[0,:,0], label='predicted suseptible')
# ax[0].scatter_density(time_day.iloc[idx_list], S_list, label='estimated suseptible', cmap=white_viridis)
# ax[0].legend()
# plt.setp(ax[0].get_xticklabels(), rotation=45)
# # ax[0].set_title(f"{country}")

    
# ax[1].plot(time_day, data_train, label='estimated infectious')
# ax[1].scatter_density(time_day[idx_list], I_list, label=f'{pred_length} days prediction', cmap=white_viridis)
# ax[1].legend()
# plt.setp(ax[1].get_xticklabels(), rotation=45)
# # ax[1].set_title(f"{country}")

# # ax[2].plot(time_day[idx_list], pred[0,:,2], label='predicted recovered')
# ax[2].scatter_density(time_day.iloc[idx_list], R_list, label='estimated recovered', cmap=white_viridis)
# ax[2].legend()
# plt.setp(ax[2].get_xticklabels(), rotation=45)
# # ax[2].set_title(f"{country}")

# fig.suptitle(f"{country} datasets")
# # fig.colorbar(density, label='Number of points per pixel')

# os.makedirs(f'./figures/{file_name}_prediction', exist_ok=True)
# fig.savefig(f'./figures/{file_name}_prediction/{country}_{pred_length}days_prediction.png', \
#             bbox_inches='tight', dpi=300)


    

import numpy as np
from scipy import stats

m1 = np.array(time_day[idx_list])
scale = m1.max()

m1 = m1/scale
m2 = np.array(I_list)

xmin = m1.min()
xmax = m1.max()
ymin = m2.min()
ymax = m2.max()

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([m1,m2])

kernel = stats.gaussian_kde(values)

Z = np.reshape(kernel(positions).T, X.shape)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()

Z_ = np.rot90(Z)
# ax.imshow(Z_, cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])

Z_norm = Z_/Z_.sum(axis=0)
ax.imshow(Z_norm, cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])

# ax.plot(m1, m2, 'k.', markersize=2)
ax.plot(time_day/scale, data_train, label='estimated infectious')

ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
plt.show()
    
# import statsmodels.api as sm

# x = sm.add_constant(x1) # adding a constant
# lm = sm.OLS(y,x).fit() # fitting the model



