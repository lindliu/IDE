

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
                
        
    path_u = glob.glob(f'../figures__/{file_name}_{start}_*')
    
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
    return path, data_train, time_day, N, file_name
    
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



### 'real Mexico', 'real South Africa', 'real Korea', 'est Mexico', 'est South Africa', 'est Korea''simulation
peak_start = np.zeros(7)#np.array([40,0,40,25,0,40,40])
peak_st = np.array([71, 41, 134, 62, 31, 120, 39]) ### first peak
peak_nd = np.array([247, 187, 283, 240, 177, 273, 178]) ### second peak

peak_st_idx, peak_nd_idx = [], []
model_st_date, model_nd_date = [], []

countries = ['Mexico', 'South Africa', 'Republic of Korea', 'Mexico', 'South Africa', 'Republic of Korea']  #'simulation'
start_list = [640, 640, 640, 640, 640, 640]
length = 400

pred_length = 7*4
xlabel = ['a', 'b', 'c', 'd', 'e', 'f']
fig, ax = plt.subplots(3,6, figsize=[40,20])
ax = ax.flatten()
for idx in range(6):
    
    country = countries[idx]
    start = start_list[idx]

    estimate, prop = (False, False) if idx<=2 else (True, True)
    filename = f'estimate_{country}' if estimate else f'real_{country}'
    
    path, data_train, time_day, N, file_name = load_result_path(country, start, estimate, prop, length)
    # pred_idx, mu_list, sigma_list, prediction_I, prediction_I_, prediction_S, prediction_R = load_results(path, pred_length, pred_length_)
    pred_idx, prediction_I, prediction_I_ = [], [], []
    prediction_S, prediction_R = [], []
    mu_list, sigma_list = [], []
    tau_list = []
    
    t_end = 25
    # T = np.linspace(0., t_end, 400)[:length][::-1]
    
    peak_st_idx.append([])
    peak_nd_idx.append([])
    model_st_date.append([])
    model_nd_date.append([])
    for pp in path:
        
        idx_end = int(pp.split('/')[-2].split('_')[-1])
        pos = idx_end-start
    
        data_pred = np.load(pp)
        pred = data_pred['pred'][:,:length,:]
            
        mu_list.append(data_pred['mu'].item())
        sigma_list.append(data_pred['sigma'].item())
        tau_list.append(data_pred['tau'].item())
        
        pred_idx.append(list(np.arange(pos, pos+pred_length))[-1])
        prediction_I.append(list(pred[0,pos:pos+pred_length,1])[-1])        
        prediction_S.append(list(pred[0,pos:pos+pred_length,0])[-1])
        prediction_R.append(list(pred[0,pos:pos+pred_length,2])[-1])
        
        ax[idx].scatter(time_day.iloc[np.arange(pos, pos+pred_length)], \
                    pred[0,pos:pos+pred_length,1]/N, c='tab:blue',alpha=.1,s=20)
        
        # if pos<peak_st[idx]-pred_length:
        if pos<peak_st[idx] and pos>peak_st[idx]-pred_length:
            peak_idx = np.argmax(pred[0,pos:pos+pred_length,1]/N)
            ax[idx].scatter(time_day.iloc[np.arange(pos, pos+pred_length)[peak_idx]], \
                        pred[0,pos+peak_idx,1]/N, c='b',alpha=.9,s=60)
            
            peak_st_idx[-1].append(peak_idx+pos)
            model_st_date[-1].append(pos)
            
        if pos<peak_nd[idx] and pos>peak_nd[idx]-pred_length:
            peak_idx = np.argmax(pred[0,pos:pos+pred_length,1]/N)
            ax[idx].scatter(time_day.iloc[np.arange(pos, pos+pred_length)[peak_idx]], \
                        pred[0,pos+peak_idx,1]/N, c='b',alpha=.9,s=60)
            
            peak_nd_idx[-1].append(peak_idx+pos)
            model_nd_date[-1].append(pos)
            
    mu_list = np.array(mu_list)
    sigma_list = np.array(sigma_list)
    
    prediction_I = np.array(prediction_I)
    prediction_S = np.array(prediction_S)
    prediction_R = np.array(prediction_R)
        
    
    ax[idx].scatter(time_day.iloc[np.arange(pos, idx_end-start+53)], \
                    pred[0,pos:pos+53,1]/N, c='tab:blue',alpha=.1,s=20)
    
    
    if estimate:
        ax[idx].plot(time_day, data_train/N, c='r', label='Estimated I')
    else:
        ax[idx].plot(time_day, data_train/N, c='r', label='Daily average I')
    ax[idx].legend()
    ax[idx].axvline(x=time_day[peak_st[idx]+start], color='r', linestyle='dashed', label='axvline')
    ax[idx].axvline(x=time_day[peak_nd[idx]+start], color='r', linestyle='dashed', label='axvline')
    ax[idx].set_ylim(0,data_train.max()/N*2)
    plt.setp(ax[idx].get_xticklabels(), rotation=45)
    ax[idx].set_xlabel(f"({xlabel[idx]}1)")
    

    ax[idx+6*1].scatter(time_day.iloc[peak_st_idx[-1]], time_day.iloc[model_st_date[-1]], c='b')
    ax[idx+6*1].scatter(time_day.iloc[peak_nd_idx[-1]], time_day.iloc[model_nd_date[-1]], c='b')
    ax[idx+6*1].set_xlim(time_day.iloc[0], time_day.iloc[399])
    ax[idx+6*1].set_ylim(time_day.iloc[0], time_day.iloc[300])
    ax[idx+6*1].axvline(x=time_day[peak_st[idx]+start], color='r', linestyle='dashed', label='axvline')
    ax[idx+6*1].axvline(x=time_day[peak_nd[idx]+start], color='r', linestyle='dashed', label='axvline')
    plt.setp(ax[idx+6*1].get_xticklabels(), rotation=45)

        
    # data_path = glob.glob(f'../figures_trend__/{filename}_{start}_*/*.npz')[0]
    data_path = pp
    endind = int(data_path.split('/')[-2].split('_')[-1])    
    ax[idx].axvline(x=time_day[endind], color='k', linestyle='dashed', label='axvline')

    ax[idx+6*2].axvline(x=time_day[endind], color='k', linestyle='dashed', label='axvline')

    data_series = np.load(data_path)
    # data_pred = data_series['pred'][0]/N
    data_beta = data_series['beta']

    ax[idx+6*2].plot(time_day, data_beta[:, 0], label='$R_0(t)$')
    ax[idx+6*2].legend()
    # ax[idx+6*2].axvline(x=time_day[start], color='k', linestyle='dashed', label='axvline')
    ax[idx+6*2].axvline(x=time_day[endind], color='k', linestyle='dashed', label='axvline')
    plt.setp(ax[idx+6*2].get_xticklabels(), rotation=45)
    ax[idx+6*2].set_xlabel(f"({xlabel[idx]}3)")



ax = ax.reshape(3,6)
pad = 5
rows = ['Mexico', 'South Africa', 'Republic of Korea']
rows = ['Infected', 'Model date', '$R_0(t)$']
for ax_, row in zip(ax[:,0], rows):
    ax_.annotate(row, xy=(0, 0.5), xytext=(-ax_.yaxis.labelpad - pad, 0),
                xycoords=ax_.yaxis.label, textcoords='offset points',
                fontsize=30, ha='right', va='center', rotation=90)

cols = [f'{pred_length}-day prediction\n(average daily cases)', 
        '$R_0(t)$\n(average daily cases)', f'{pred_length}-day prediction\n(estimated datasets)', 
        '$R_0(t)$\n(estimated datasets)']
cols = ['Mexico\n(average daily cases)', 'South Africa\n(average daily cases)', 'Republic of Korea\n(average daily cases)',\
        'Mexico\n(estimated cases)', 'South Africa\n(estimated cases)', 'Republic of Korea\n(estimated cases)']
for ax_, col in zip(ax[0], cols):
    ax_.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                fontsize=30, ha='center', va='baseline')

fig.tight_layout()
fig.savefig(f'./prediction_trend_peak.png', \
            bbox_inches='tight', dpi=300)


    
    

# peak_st = np.array([71, 41, 134, 62, 31, 120, 39]) ### first peak
# peak_nd = np.array([247, 187, 283, 240, 177, 273, 178]) ### second peak

print('First peak:')
error_M_R = np.median(abs(np.array(peak_st_idx[0])-peak_st[0]))
print(f'Real Mexico: {error_M_R}')
error_M_E = np.median(abs(np.array(peak_st_idx[1])-peak_st[3]))
print(f'Estimated Mexico: {error_M_E}')
error_SA_R = np.median(abs(np.array(peak_st_idx[2])-peak_st[1]))
print(f'Real SA: {error_SA_R}')
error_SA_E = np.median(abs(np.array(peak_st_idx[3])-peak_st[4]))
print(f'Estimated SA: {error_SA_E}')
error_SK_R = np.median(abs(np.array(peak_st_idx[4])-peak_st[2]))
print(f'Real SK: {error_SK_R}')
error_SK_E = np.median(abs(np.array(peak_st_idx[5])-peak_st[5]))
print(f'Estimated SK: {error_SK_E}')


print('Second peak:')
error_M_R = np.median(abs(np.array(peak_nd_idx[0])-peak_nd[0]))
print(f'Real Mexico: {error_M_R}')
error_M_E = np.median(abs(np.array(peak_nd_idx[1])-peak_nd[3]))
print(f'Estimated Mexico: {error_M_E}')
error_SA_R = np.median(abs(np.array(peak_nd_idx[2])-peak_nd[1]))
print(f'Real SA: {error_SA_R}')
error_SA_E = np.median(abs(np.array(peak_nd_idx[3])-peak_nd[4]))
print(f'Estimated SA: {error_SA_E}')
error_SK_R = np.median(abs(np.array(peak_nd_idx[4])-peak_nd[2]))
print(f'Real SK: {error_SK_R}')
error_SK_E = np.median(abs(np.array(peak_nd_idx[5])-peak_nd[5]))
print(f'Estimated SK: {error_SK_E}')



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
                
        
    path_u = glob.glob(f'../figures__/{file_name}_{start}_*')
    
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



# start = 0

# estimate, prop = (False, False) if idx<=2 else (True, True)
# filename = f'estimate_{country}' if estimate else f'real_{country}'

# path, data_train, time_day, N, file_name = load_result_path(country, start, estimate, prop, length)
pred_length = 7*5
# pred_idx, mu_list, sigma_list, prediction_I, prediction_I_, prediction_S, prediction_R = load_results(path, pred_length, pred_length_)
pred_idx, prediction_I = [], []
prediction_S, prediction_R = [], []
mu_list, sigma_list = [], []
tau_list = []

t_end = 25
# T = np.linspace(0., t_end, 400)[:length][::-1]
peak_st_sim_idx, peak_nd_sim_idx, peak_rd_sim_idx = [], [], []
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
    prediction_S.append(list(pred[0,pos:pos+pred_length,0])[-1])
    prediction_R.append(list(pred[0,pos:pos+pred_length,2])[-1])
    
    ax[4].scatter(time_day.iloc[np.arange(pos,pos+pred_length)], \
                pred[0,pos:pos+pred_length,1]/N, c='tab:blue',alpha=.1,s=30)

    # if pos<peak_st[idx]-pred_length:
    if pos<39 and pos>0:
        peak_idx = np.argmax(pred[0,pos:pos+pred_length,1]/N)
        ax[4].scatter(time_day.iloc[np.arange(pos,pos+pred_length)[peak_idx]], \
                    pred[0,pos+peak_idx,1]/N, c='b',alpha=.9,s=60)
    
        peak_st_sim_idx.append(peak_idx+pos)

    if pos<178 and pos>178-pred_length:
        peak_idx = np.argmax(pred[0,pos:pos+pred_length,1]/N)
        ax[4].scatter(time_day.iloc[np.arange(pos,pos+pred_length)[peak_idx]], \
                    pred[0,pos+peak_idx,1]/N, c='b',alpha=.9,s=60)
    
        peak_nd_sim_idx.append(peak_idx+pos)

    if pos<291 and pos>291-pred_length:
        peak_idx = np.argmax(pred[0,pos:pos+pred_length,1]/N)
        ax[4].scatter(time_day.iloc[np.arange(pos,pos+pred_length)[peak_idx]], \
                    pred[0,pos+peak_idx,1]/N, c='b',alpha=.9,s=60)
    
        peak_rd_sim_idx.append(peak_idx+pos)

                
ax[4].scatter(time_day.iloc[np.arange(pos, pos+53)], \
            pred[0,pos:pos+53,1]/N, c='tab:blue',alpha=.1,s=30)

mu_list = np.array(mu_list)
sigma_list = np.array(sigma_list)

prediction_I = np.array(prediction_I)
prediction_S = np.array(prediction_S)
prediction_R = np.array(prediction_R)

endind = int(pp.split('/')[-2].split('_')[-1])    

ax[4].plot(time_day, data_train/N, c='r', label='Synthetic I')
ax[4].legend()
ax[4].axvline(x=time_day[357], color='k', linestyle='dashed', label='axvline')
plt.setp(ax[4].get_xticklabels(), rotation=45)
ax[4].set_xlabel(f"(c)")


ax[5].plot(time_day, data['S']/N, c='r', label='Synthetic S')
ax[5].plot(time_day, data_pred['pred'][0,:,0], label='S')
ax[5].legend()
ax[5].axvline(x=time_day[endind], color='k', linestyle='dashed', label='axvline')
plt.setp(ax[5].get_xticklabels(), rotation=45)
ax[5].set_xlabel(f"(d)")

ax[6].plot(time_day, data['R']/N, c='r', label='Synthetic R')
ax[6].plot(time_day, data_pred['pred'][0,:,2], label='R')
ax[6].legend()
ax[6].axvline(x=time_day[endind], color='k', linestyle='dashed', label='axvline')
plt.setp(ax[6].get_xticklabels(), rotation=45)
ax[6].set_xlabel(f"(e)")


ax[7].plot(time_day, data_pred['beta'], label='$R_0(t)$')
ax[7].legend()
ax[7].axvline(x=time_day[endind], color='k', linestyle='dashed', label='axvline')

beta_real = 2.3
ax[7].axhline(y=beta_real, color='r', linestyle='dashed', label='axvline')
ax[7].set_yticks(list(np.arange(1,4.5,.5)) + [beta_real])
plt.setp(ax[7].get_xticklabels(), rotation=45)
ax[7].set_xlabel(f"(f)")


# fig.tight_layout()
# fig.savefig(f'./prediction_trend_synthetic__.png', \
#             bbox_inches='tight', dpi=300)

    


print('Second peak:')
error_S_st = np.median(abs(np.array(peak_st_sim_idx)-39))
print(f'Simulation: {error_S_st}')

print('Second peak:')
error_S_nd = np.median(abs(np.array(peak_nd_sim_idx)-178))
print(f'Simulation: {error_S_nd}')

print('Third peak:')
error_S_rd = np.median(abs(np.array(peak_rd_sim_idx)-291))
print(f'Simulation: {error_S_rd}')
