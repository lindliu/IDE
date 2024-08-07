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


directory = 'samples_'
    
    

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
fig, ax = plt.subplots(3, 3, figsize=[18,12], gridspec_kw={'height_ratios': [1, 1, 1]})
fontsize = 18
fontsize_label = 13

countries = ['real_Mexico', 'real_South Africa', 'real_Republic of Korea']


idx = 0
country = countries[idx]
path = glob.glob(f'{directory}/{country}*')
path = sorted(path, key=lambda x: int(x.split('_')[-1]))
path = [glob.glob(os.path.join(path_, '*.npz'))[0] for path_ in path]

K = []
mu, sigma, beta, pred = [], [], [], []
for path_ in path:
    data = np.load(path_)
    
    K.append(data['K'].flatten())
    mu.append(data['mu'].item())
    sigma.append(data['sigma'].item())
    beta.append(data['beta'].flatten())
    pred.append(data['pred'][0])
    
mu = np.array(mu)
sigma = np.array(sigma)
beta = np.array(beta).T
pred = np.stack(pred)

from scipy.stats import norm
t = np.linspace(0,347,347*10)
distribution = []
for mean, sd in zip(mu,sigma):
    dist = norm.pdf(t, loc=mean, scale=sd)
    
    norm_term = norm.pdf(np.linspace(0,480,480*10), loc=mean, scale=sd)
    norm_term = norm_term[1:].sum()*np.linspace(0,480,480*10)[1]

    dist = dist/norm_term
    # dist = dist/(dist.sum()*(t[1]-t[0]))
    distribution.append(dist[::-1])
    # plt.plot(t,dist[::-1])

distribution=np.array(distribution).T
# plt.plot(t, distribution)


ax[idx,0].plot(t,distribution, alpha=.5)
mu_median = round(np.median(mu))
ax[idx,0].axvline(347-mu_median, color='purple', label='axvline')
ax[idx,0].set_xticks([347-mu_median,347], [mu_median,0])
# ax[idx,0].set_yticks([])
ax[idx,0].tick_params(axis='both', which='major', labelsize=fontsize_label)
ax[idx,0].set_xlabel("(a1)",fontsize=fontsize)

ax_inset1 = inset_axes(ax[idx,0], width="20%", height="70%", loc='upper right',\
                       bbox_to_anchor=(-.33, -.07, 1, 1), bbox_transform=ax[idx,0].transAxes)

violin_parts = ax_inset1.violinplot(mu, showmeans=False, showmedians=True, widths=.1)
violin_parts['cmedians'].set_color('purple')
ax_inset1.set_xticks([])#[1], [r'$\mu$'])
ax_inset1.set_title(r'$\mu$', fontsize=fontsize)
# ax_inset1.set_yticks([50,60,70,74,80])
ax_inset1.tick_params(axis='both', which='major', labelsize=fontsize_label)
# for pc in violin_parts['bodies']:
#     pc.set_facecolor('red')
#     pc.set_edgecolor('black')
# x = np.random.normal(1, 0.01, size=len(mu_10))
# x[mu_10.argmin()]=1
# from matplotlib import cm
# ax_inset1.scatter(x,mu_10, c=cm.prism(3), alpha=0.2, s=20)

ax_inset2 = inset_axes(ax[idx,0], width="20%", height="70%", loc='upper right',\
                       bbox_to_anchor=(.0, -.07, 1, 1), bbox_transform=ax[idx,0].transAxes)

violin_parts = ax_inset2.violinplot(sigma, showmeans=False, showmedians=True, widths=.1)
violin_parts['cmedians'].set_color('purple')
ax_inset2.set_xticks([])#[1], [r'$\sigma$'])
ax_inset2.set_title(r'$\sigma$', fontsize=fontsize)
# ax_inset2.set_yticks([1,10,20,30])
ax_inset2.tick_params(axis='both', which='major', labelsize=fontsize_label)
# ax_inset2.legend(fontsize=5)




ax[idx,1].fill_between(np.arange(0, 400), np.min(beta, axis=1), np.max(beta, axis=1),
        alpha=0.2, facecolor='#089FFF', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax[idx,1].plot(np.median(beta,axis=1), label=r'Median $R_0(t)$')
ax[idx,1].axvline(347, color='gray', linestyle='dashed')#, label='axvline')
ax[idx,1].axhline(y=1, color='gray', linestyle='dashed')
ax[idx,1].set_xticks([0,200,347,400])
ax[idx,1].set_yticks([1,3,5,7,9])
ax[idx,1].tick_params(axis='both', which='major', labelsize=fontsize_label)
ax[idx,1].set_xlabel("(a2)",fontsize=fontsize)
ax[idx,1].set_ylim(0,12.3)
ax[idx,1].legend(fontsize=fontsize, loc='upper left')



data = pd.read_csv(f'../../data/covid_Mexico.csv', sep='\t')
data['date'] = pd.to_datetime(data['date'])#['I'][640:640+400].to_numpy().reshape([1,-1,1])   
population = data['population'][0]
data_ = get_train_data(data, start=640, length=400, recovery_time=14, estimate=False, prop=False)

ax[idx,2].plot(data_[0,:,0]/population, c='r', label='Daily average j')
ax[idx,2].fill_between(np.arange(0, 400), np.min(pred[:,:,1], axis=0)/population, np.max(pred[:,:,1], axis=0)/population,
        alpha=0.2, facecolor='#089FFF', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax[idx,2].plot(np.median(pred[:,:,1],axis=0)/population, label=r'Median j')
ax[idx,2].axvline(347, color='gray', linestyle='dashed')#, label='axvline')
ax[idx,2].set_xticks([0,200,347,400])
# ax[idx,2].set_yticks([2,4,6,8])
ax[idx,2].tick_params(axis='both', which='major', labelsize=fontsize_label)
ax[idx,2].set_xlabel("(a3)",fontsize=fontsize)
ax[idx,2].set_ylim(0, ax[idx,2].axes.get_ylim()[1]*1.3)
ax[idx,2].legend(fontsize=fontsize)



R_e = pred[:,:,0].T*beta/population
ax[idx,1].fill_between(np.arange(0, 400), np.min(R_e, axis=1), np.max(R_e, axis=1),
        alpha=0.2, facecolor='#F39C12', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax[idx,1].plot(np.median(R_e, axis=1), label=r'Median $R_e$')
ax[idx,1].legend(fontsize=fontsize)








idx = 1
country = countries[idx]
path = glob.glob(f'{directory}/{country}*')
path = sorted(path, key=lambda x: int(x.split('_')[-1]))
path = [glob.glob(os.path.join(path_, '*.npz'))[0] for path_ in path]

K = []
mu, sigma, beta, pred = [], [], [], []
for path_ in path:
    data = np.load(path_)
    
    K.append(data['K'].flatten())
    mu.append(data['mu'].item())
    sigma.append(data['sigma'].item())
    beta.append(data['beta'].flatten())
    pred.append(data['pred'][0])
    
mu = np.array(mu)
sigma = np.array(sigma)
beta = np.array(beta).T
pred = np.stack(pred)


from scipy.stats import norm
t = np.linspace(0,347,347*10)
distribution = []
for mean, sd in zip(mu,sigma):
    dist = norm.pdf(t, loc=mean, scale=sd)
    
    norm_term = norm.pdf(np.linspace(0,480,480*10), loc=mean, scale=sd)
    norm_term = norm_term[1:].sum()*np.linspace(0,480,480*10)[1]

    dist = dist/norm_term
    # dist = dist/(dist.sum()*(t[1]-t[0]))
    distribution.append(dist[::-1])
    # plt.plot(t,dist[::-1])

distribution=np.array(distribution).T
# plt.plot(t, distribution)



ax[idx,0].plot(t,distribution, alpha=.5)
mu_median = round(np.median(mu))
ax[idx,0].axvline(347-mu_median, color='purple', label='axvline')
ax[idx,0].set_xticks([347-mu_median,347], [mu_median,0])
# ax[idx,0].set_yticks([])
ax[idx,0].tick_params(axis='both', which='major', labelsize=fontsize_label)
ax[idx,0].set_xlabel("(b1)",fontsize=fontsize)

ax_inset1 = inset_axes(ax[idx,0], width="20%", height="70%", loc='upper left',\
                       bbox_to_anchor=(.39, -.07, 1, 1), bbox_transform=ax[idx,0].transAxes)

violin_parts = ax_inset1.violinplot(mu, showmeans=False, showmedians=True, widths=.1)
violin_parts['cmedians'].set_color('purple')
ax_inset1.set_xticks([])#[1], [r'$\mu$'])
ax_inset1.set_title(r'$\mu$', fontsize=fontsize)
# ax_inset1.set_yticks([50,60,70,74,80])
ax_inset1.tick_params(axis='both', which='major', labelsize=fontsize_label)


ax_inset2 = inset_axes(ax[idx,0], width="20%", height="70%", loc='upper left',\
                       bbox_to_anchor=(.72, -.07, 1, 1), bbox_transform=ax[idx,0].transAxes)

violin_parts = ax_inset2.violinplot(sigma, showmeans=False, showmedians=True, widths=.1)
violin_parts['cmedians'].set_color('purple')
ax_inset2.set_xticks([])#[1], [r'$\sigma$'])
ax_inset2.set_title(r'$\sigma$', fontsize=fontsize)
# ax_inset2.set_yticks([1,10,20,30])
ax_inset2.tick_params(axis='both', which='major', labelsize=fontsize_label)
# ax_inset2.legend(fontsize=5)



ax[idx,1].fill_between(np.arange(0, 400), np.min(beta, axis=1), np.max(beta, axis=1),
        alpha=0.2, facecolor='#089FFF', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax[idx,1].plot(np.median(beta,axis=1), label=r'Median $R_0(t)$')
ax[idx,1].axvline(347, color='gray', linestyle='dashed')#, label='axvline')
ax[idx,1].axhline(y=1, color='gray', linestyle='dashed')
ax[idx,1].set_xticks([0,200,347,400])
ax[idx,1].set_yticks([1,3,5,7,9])
ax[idx,1].tick_params(axis='both', which='major', labelsize=fontsize_label)
ax[idx,1].set_xlabel("(b2)",fontsize=fontsize)
ax[idx,1].set_ylim(0,12)
ax[idx,1].legend(fontsize=fontsize)




data = pd.read_csv(f'../../data/covid_South Africa.csv', sep='\t')
data['date'] = pd.to_datetime(data['date'])#['I'][640:640+400].to_numpy().reshape([1,-1,1])   
population = data['population'][0]
data_ = get_train_data(data, start=640, length=400, recovery_time=14, estimate=False, prop=False)

ax[idx,2].plot(data_[0,:,0]/population, c='r', label='Daily average j')
ax[idx,2].fill_between(np.arange(0, 400), np.min(pred[:,:,1], axis=0)/population, np.max(pred[:,:,1], axis=0)/population,
        alpha=0.2, facecolor='#089FFF', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax[idx,2].plot(np.median(pred[:,:,1],axis=0)/population, label=r'Median j')
ax[idx,2].axvline(347, color='gray', linestyle='dashed')#, label='axvline')
ax[idx,2].set_xticks([0,200,347,400])
# ax[idx,2].set_yticks([2,4,6,8])
ax[idx,2].tick_params(axis='both', which='major', labelsize=fontsize_label)
ax[idx,2].set_xlabel("(b3)",fontsize=fontsize)
ax[idx,2].set_ylim(0, ax[idx,2].axes.get_ylim()[1]*1.3)
ax[idx,2].legend(fontsize=fontsize)


R_e = pred[:,:,0].T*beta/population
ax[idx,1].fill_between(np.arange(0, 400), np.min(R_e, axis=1), np.max(R_e, axis=1),
        alpha=0.2, facecolor='#F39C12', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax[idx,1].plot(np.median(R_e, axis=1), label=r'Median $R_e$')
ax[idx,1].legend(fontsize=fontsize)







idx = 2
country = countries[idx]
path = glob.glob(f'{directory}/{country}*')
path = sorted(path, key=lambda x: int(x.split('_')[-1]))
path = [glob.glob(os.path.join(path_, '*.npz'))[0] for path_ in path]

K = []
mu, sigma, beta, pred = [], [], [], []
for path_ in path:
    data = np.load(path_)
    
    K.append(data['K'].flatten())
    mu.append(data['mu'].item())
    sigma.append(data['sigma'].item())
    beta.append(data['beta'].flatten())
    pred.append(data['pred'][0])
    
mu = np.array(mu)
sigma = np.array(sigma)
beta = np.array(beta).T
pred = np.stack(pred)


from scipy.stats import norm
t = np.linspace(0,347,347*10)
distribution = []
for mean, sd in zip(mu,sigma):
    dist = norm.pdf(t, loc=mean, scale=sd)
    
    norm_term = norm.pdf(np.linspace(0,480,480*10), loc=mean, scale=sd)
    norm_term = norm_term[1:].sum()*np.linspace(0,480,480*10)[1]

    dist = dist/norm_term
    # dist = dist/(dist.sum()*(t[1]-t[0]))
    distribution.append(dist[::-1])
    # plt.plot(t,dist[::-1])

distribution=np.array(distribution).T
# plt.plot(t, distribution)



ax[idx,0].plot(t,distribution, alpha=.5)
mu_median = round(np.median(mu))
ax[idx,0].axvline(347-mu_median, color='purple', label='axvline')
ax[idx,0].set_xticks([0,347-mu_median,347], [347,mu_median,0])
# ax[idx,0].set_yticks([])
ax[idx,0].tick_params(axis='both', which='major', labelsize=11)
ax[idx,0].set_xlabel("(c1)",fontsize=fontsize)
ax[idx,0].set_ylim(0,0.1)

ax_inset1 = inset_axes(ax[idx,0], width="20%", height="70%", loc='upper left',\
                       bbox_to_anchor=(.23, -.07, 1, 1), bbox_transform=ax[idx,0].transAxes)

violin_parts = ax_inset1.violinplot(mu, showmeans=False, showmedians=True, widths=.1)
violin_parts['cmedians'].set_color('purple')
ax_inset1.set_xticks([])#[1], [r'$\mu$'])
ax_inset1.set_title(r'$\mu$', fontsize=fontsize)
# ax_inset1.set_yticks([50,60,70,74,80])
ax_inset1.tick_params(axis='both', which='major', labelsize=fontsize_label)


ax_inset2 = inset_axes(ax[idx,0], width="20%", height="70%", loc='upper left',\
                       bbox_to_anchor=(.73, -.07, 1, 1), bbox_transform=ax[idx,0].transAxes)

violin_parts = ax_inset2.violinplot(sigma, showmeans=False, showmedians=True, widths=.1)
violin_parts['cmedians'].set_color('purple')
ax_inset2.set_xticks([])#[1], [r'$\sigma$'])
ax_inset2.set_title(r'$\sigma$', fontsize=fontsize)
# ax_inset2.set_yticks([1,10,20,30])
ax_inset2.tick_params(axis='both', which='major', labelsize=fontsize_label)
# ax_inset2.legend(fontsize=5)



ax[idx,1].fill_between(np.arange(0, 400), np.min(beta, axis=1), np.max(beta, axis=1),
        alpha=0.2, facecolor='#089FFF', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax[idx,1].plot(np.median(beta,axis=1), label=r'Median $R_0(t)$')
ax[idx,1].axvline(347, color='gray', linestyle='dashed')#, label='axvline')
ax[idx,1].axhline(y=1, color='gray', linestyle='dashed')
ax[idx,1].set_xticks([0,200,347,400])
ax[idx,1].set_yticks([1,3,5,7,9])
ax[idx,1].tick_params(axis='both', which='major', labelsize=fontsize_label)
ax[idx,1].set_xlabel("(c2)",fontsize=fontsize)
ax[idx,1].set_ylim(0,12)
ax[idx,1].legend(fontsize=fontsize)


data = pd.read_csv(f'../../data/covid_Republic of Korea.csv', sep='\t')
data['date'] = pd.to_datetime(data['date'])#['I'][640:640+400].to_numpy().reshape([1,-1,1])   
population = data['population'][0]
data_ = get_train_data(data, start=640, length=400, recovery_time=14, estimate=False, prop=False)

ax[idx,2].plot(data_[0,:,0]/population, c='r', label='Daily average j')
ax[idx,2].fill_between(np.arange(0, 400), np.min(pred[:,:,1], axis=0)/population, np.max(pred[:,:,1], axis=0)/population,
        alpha=0.2, facecolor='#089FFF', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax[idx,2].plot(np.median(pred[:,:,1],axis=0)/population, label=r'Median j')
ax[idx,2].axvline(347, color='gray', linestyle='dashed')#, label='axvline')
ax[idx,2].set_xticks([0,200,347,400])
# ax[idx,2].set_yticks([2,4,6,8])
ax[idx,2].tick_params(axis='both', which='major', labelsize=fontsize_label)
ax[idx,2].set_xlabel("(c3)",fontsize=fontsize)
ax[idx,2].set_ylim(0, ax[2,idx].axes.get_ylim()[1]*1.3)
ax[idx,2].legend(fontsize=fontsize)


R_e = pred[:,:,0].T*beta/population
ax[idx,1].fill_between(np.arange(0, 400), np.min(R_e, axis=1), np.max(R_e, axis=1),
        alpha=0.2, facecolor='#F39C12', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax[idx,1].plot(np.median(R_e, axis=1), label=r'Median $R_e$')
ax[idx,1].legend(fontsize=fontsize)




pad = 5
# rows = ['Mexico\n(average daily cases)', 'South Africa\n(average daily cases)', 'South Korea\n(average daily cases)']
rows = ['Mexico', 'South Africa', 'South Korea']
for ax_, row in zip(ax[:,0], rows):
    ax_.annotate(row, xy=(0, 0.5), xytext=(-ax_.yaxis.labelpad - pad, 0),
                xycoords=ax_.yaxis.label, textcoords='offset points',
                fontsize=25, ha='center', va='center', rotation=90)
    
cols = ['$K(\cdot)$', '$R_0(t)$', 'j']
for ax_, col in zip(ax[0], cols):
    ax_.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                fontsize=25, ha='center', va='baseline')

# fig.tight_layout()
fig.savefig('./figures/samples_actual_real.png', bbox_inches='tight', dpi=300)














from mpl_toolkits.axes_grid1.inset_locator import inset_axes
fig, ax = plt.subplots(3, 3, figsize=[18,12], gridspec_kw={'height_ratios': [1, 1, 1]})
fontsize = 18

countries = ['estimate_Mexico', 'estimate_South Africa', 'estimate_Republic of Korea']




idx = 0
country = countries[idx]
path = glob.glob(f'{directory}/{country}*')
path = sorted(path, key=lambda x: int(x.split('_')[-1]))
path = [glob.glob(os.path.join(path_, '*.npz'))[0] for path_ in path]

K = []
mu, sigma, beta, pred = [], [], [], []
for path_ in path:
    data = np.load(path_)
    
    K.append(data['K'].flatten())
    mu.append(data['mu'].item())
    sigma.append(data['sigma'].item())
    beta.append(data['beta'].flatten())
    pred.append(data['pred'][0])
    
mu = np.array(mu)
sigma = np.array(sigma)
beta = np.array(beta).T
pred = np.stack(pred)


from scipy.stats import norm
t = np.linspace(0,347,347*10)
distribution = []
for mean, sd in zip(mu,sigma):
    dist = norm.pdf(t, loc=mean, scale=sd)
    
    norm_term = norm.pdf(np.linspace(0,480,480*10), loc=mean, scale=sd)
    norm_term = norm_term[1:].sum()*np.linspace(0,480,480*10)[1]

    dist = dist/norm_term
    # dist = dist/(dist.sum()*(t[1]-t[0]))
    distribution.append(dist[::-1])
    # plt.plot(t,dist[::-1])

distribution=np.array(distribution).T
# plt.plot(t, distribution)



ax[idx,0].plot(t,distribution, alpha=.5)
mu_median = round(np.median(mu))
ax[idx,0].axvline(347-mu_median, color='purple', label='axvline')
ax[idx,0].set_xticks([0,347-mu_median,347], [347,mu_median,0])
# ax[idx,0].set_yticks([])
ax[idx,0].tick_params(axis='both', which='major', labelsize=fontsize_label)
ax[idx,0].set_xlabel("(d1)",fontsize=fontsize)

ax_inset1 = inset_axes(ax[idx,0], width="18%", height="70%", loc='upper left',\
                       bbox_to_anchor=(.12, -.07, 1, 1), bbox_transform=ax[idx,0].transAxes)

violin_parts = ax_inset1.violinplot(mu, showmeans=False, showmedians=True, widths=.1)
violin_parts['cmedians'].set_color('purple')
ax_inset1.set_xticks([])#[1], [r'$\mu$'])
ax_inset1.set_title(r'$\mu$', fontsize=fontsize)
# ax_inset1.set_yticks([50,60,70,74,80])
ax_inset1.tick_params(axis='both', which='major', labelsize=fontsize_label)


ax_inset2 = inset_axes(ax[idx,0], width="18%", height="70%", loc='upper left',\
                       bbox_to_anchor=(.43, -.07, 1, 1), bbox_transform=ax[idx,0].transAxes)

violin_parts = ax_inset2.violinplot(sigma, showmeans=False, showmedians=True, widths=.1)
violin_parts['cmedians'].set_color('purple')
ax_inset2.set_xticks([])#[1], [r'$\sigma$'])
ax_inset2.set_title(r'$\sigma$', fontsize=fontsize)
# ax_inset2.set_yticks([1,10,20,30])
ax_inset2.tick_params(axis='both', which='major', labelsize=fontsize_label)
# ax_inset2.legend(fontsize=5)



ax[idx,1].fill_between(np.arange(0, 400), np.min(beta, axis=1), np.max(beta, axis=1),
        alpha=0.2, facecolor='#089FFF', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax[idx,1].plot(np.median(beta,axis=1), label=r'Median $R_0(t)$')
ax[idx,1].axvline(347, color='gray', linestyle='dashed')#, label='axvline')
ax[idx,1].axhline(y=1, color='gray', linestyle='dashed')
ax[idx,1].set_xticks([0,200,347,400])
ax[idx,1].set_yticks([1,3,5,7,9])
ax[idx,1].tick_params(axis='both', which='major', labelsize=fontsize_label)
ax[idx,1].set_xlabel("(d2)",fontsize=fontsize)
ax[idx,1].set_ylim(0,12)
ax[idx,1].legend(fontsize=fontsize)


data = pd.read_csv(f'../../data/covid_Mexico.csv', sep='\t')
data['date'] = pd.to_datetime(data['date'])#['I'][640:640+400].to_numpy().reshape([1,-1,1])   
population = 1#data['population'][0]
data_ = get_train_data(data, start=640, length=400, recovery_time=14, estimate=True, prop=True)

ax[idx,2].plot(data_[0,:,0]/population, c='r', label='Estimated j')
ax[idx,2].fill_between(np.arange(0, 400), np.min(pred[:,:,1], axis=0)/population, np.max(pred[:,:,1], axis=0)/population,
        alpha=0.2, facecolor='#089FFF', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax[idx,2].plot(np.median(pred[:,:,1],axis=0)/population, label=r'Median j')
ax[idx,2].axvline(347, color='gray', linestyle='dashed')#, label='axvline')
ax[idx,2].set_xticks([0,200,347,400])
# ax[idx,2].set_yticks([2,4,6,8])
ax[idx,2].tick_params(axis='both', which='major', labelsize=fontsize_label)
ax[idx,2].set_xlabel("(d3)",fontsize=fontsize)
ax[idx,2].set_ylim(0, ax[idx,2].axes.get_ylim()[1]*1.3)
ax[idx,2].legend(fontsize=fontsize)


R_e = pred[:,:,0].T*beta/population
ax[idx,1].fill_between(np.arange(0, 400), np.min(R_e, axis=1), np.max(R_e, axis=1),
        alpha=0.2, facecolor='#F39C12', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax[idx,1].plot(np.median(R_e, axis=1), label=r'Median $R_e$')
ax[idx,1].legend(fontsize=fontsize)







idx = 1
country = countries[idx]
path = glob.glob(f'{directory}/{country}*')
path = sorted(path, key=lambda x: int(x.split('_')[-1]))
path = [glob.glob(os.path.join(path_, '*.npz'))[0] for path_ in path]

K = []
mu, sigma, beta, pred = [], [], [], []
for path_ in path:
    data = np.load(path_)
    
    K.append(data['K'].flatten())
    mu.append(data['mu'].item())
    sigma.append(data['sigma'].item())
    beta.append(data['beta'].flatten())
    pred.append(data['pred'][0])
    
mu = np.array(mu)
sigma = np.array(sigma)
beta = np.array(beta).T
pred = np.stack(pred)


from scipy.stats import norm
t = np.linspace(0,347,347*10)
distribution = []
for mean, sd in zip(mu,sigma):
    dist = norm.pdf(t, loc=mean, scale=sd)
    
    norm_term = norm.pdf(np.linspace(0,480,480*10), loc=mean, scale=sd)
    norm_term = norm_term[1:].sum()*np.linspace(0,480,480*10)[1]

    dist = dist/norm_term
    # dist = dist/(dist.sum()*(t[1]-t[0]))
    distribution.append(dist[::-1])
    # plt.plot(t,dist[::-1])

distribution=np.array(distribution).T
# plt.plot(t, distribution)



ax[idx,0].plot(t,distribution, alpha=.5)
mu_median = round(np.median(mu))
ax[idx,0].axvline(347-mu_median, color='purple', label='axvline')
ax[idx,0].set_xticks([0,347-mu_median,347], [347,mu_median,0])
# ax[idx,0].set_yticks([])
ax[idx,0].tick_params(axis='both', which='major', labelsize=fontsize_label)
ax[idx,0].set_xlabel("(e1)",fontsize=fontsize)

ax_inset1 = inset_axes(ax[idx,0], width="18%", height="70%", loc='upper left',\
                       bbox_to_anchor=(.1, -.07, 1, 1), bbox_transform=ax[idx,0].transAxes)

violin_parts = ax_inset1.violinplot(mu, showmeans=False, showmedians=True, widths=.1)
violin_parts['cmedians'].set_color('purple')
ax_inset1.set_xticks([])#[1], [r'$\mu$'])
ax_inset1.set_title(r'$\mu$', fontsize=fontsize)
# ax_inset1.set_yticks([50,60,70,74,80])
ax_inset1.tick_params(axis='both', which='major', labelsize=fontsize_label)


ax_inset2 = inset_axes(ax[idx,0], width="18%", height="70%", loc='upper left',\
                       bbox_to_anchor=(.4, -.07, 1, 1), bbox_transform=ax[idx,0].transAxes)

violin_parts = ax_inset2.violinplot(sigma, showmeans=False, showmedians=True, widths=.1)
violin_parts['cmedians'].set_color('purple')
ax_inset2.set_xticks([])#[1], [r'$\sigma$'])
ax_inset2.set_title(r'$\sigma$', fontsize=fontsize)
ax_inset2.set_yticks([25,30,35])
ax_inset2.tick_params(axis='both', which='major', labelsize=fontsize_label)
# ax_inset2.legend(fontsize=5)



ax[idx,1].fill_between(np.arange(0, 400), np.min(beta, axis=1), np.max(beta, axis=1),
        alpha=0.2, facecolor='#089FFF', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax[idx,1].plot(np.median(beta,axis=1), label=r'Median $R_0(t)$')
ax[idx,1].axvline(347, color='gray', linestyle='dashed')#, label='axvline')
ax[idx,1].axhline(y=1, color='gray', linestyle='dashed')
ax[idx,1].set_xticks([0,200,347,400])
ax[idx,1].set_yticks([1,3,5,7,9])
ax[idx,1].tick_params(axis='both', which='major', labelsize=fontsize_label)
ax[idx,1].set_xlabel("(e2)",fontsize=fontsize)
ax[idx,1].set_ylim(0,12)
ax[idx,1].legend(fontsize=fontsize)


data = pd.read_csv(f'../../data/covid_South Africa.csv', sep='\t')
data['date'] = pd.to_datetime(data['date'])#['I'][640:640+400].to_numpy().reshape([1,-1,1])   
population = 1#data['population'][0]
data_ = get_train_data(data, start=640, length=400, recovery_time=14, estimate=True, prop=True)

ax[idx,2].plot(data_[0,:,0]/population, c='r', label='Estimated j')
ax[idx,2].fill_between(np.arange(0, 400), np.min(pred[:,:,1], axis=0)/population, np.max(pred[:,:,1], axis=0)/population,
        alpha=0.2, facecolor='#089FFF', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax[idx,2].plot(np.median(pred[:,:,1],axis=0)/population, label=r'Median j')
ax[idx,2].axvline(347, color='gray', linestyle='dashed')#, label='axvline')
ax[idx,2].set_xticks([0,200,347,400])
# ax[idx,2].set_yticks([2,4,6,8])
ax[idx,2].tick_params(axis='both', which='major', labelsize=fontsize_label)
ax[idx,2].set_xlabel("(e3)",fontsize=fontsize)
ax[idx,2].set_ylim(0, ax[idx,2].axes.get_ylim()[1]*1.3)
ax[idx,2].legend(fontsize=fontsize)


R_e = pred[:,:,0].T*beta/population
ax[idx,1].fill_between(np.arange(0, 400), np.min(R_e, axis=1), np.max(R_e, axis=1),
        alpha=0.2, facecolor='#F39C12', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax[idx,1].plot(np.median(R_e, axis=1), label=r'Median $R_e$')
ax[idx,1].legend(fontsize=fontsize)






idx = 2
country = countries[idx]
path = glob.glob(f'{directory}/{country}*')
path = sorted(path, key=lambda x: int(x.split('_')[-1]))
path = [glob.glob(os.path.join(path_, '*.npz'))[0] for path_ in path]

K = []
mu, sigma, beta, pred = [], [], [], []
for path_ in path:
    data = np.load(path_)
    
    K.append(data['K'].flatten())
    mu.append(data['mu'].item())
    sigma.append(data['sigma'].item())
    beta.append(data['beta'].flatten())
    pred.append(data['pred'][0])
    
mu = np.array(mu)
sigma = np.array(sigma)
beta = np.array(beta).T
pred = np.stack(pred)


from scipy.stats import norm
t = np.linspace(0,347,347*10)
distribution = []
for mean, sd in zip(mu,sigma):
    dist = norm.pdf(t, loc=mean, scale=sd)
    
    norm_term = norm.pdf(np.linspace(0,480,480*10), loc=mean, scale=sd)
    norm_term = norm_term[1:].sum()*np.linspace(0,480,480*10)[1]

    dist = dist/norm_term
    # dist = dist/(dist.sum()*(t[1]-t[0]))
    distribution.append(dist[::-1])
    # plt.plot(t,dist[::-1])

distribution=np.array(distribution).T
# plt.plot(t, distribution)



ax[idx,0].plot(t,distribution, alpha=.5)
mu_median = round(np.median(mu))
ax[idx,0].axvline(347-mu_median, color='purple', label='axvline')
ax[idx,0].set_xticks([0,347-mu_median,347], [347,mu_median,0])
# ax[idx,0].set_yticks([])
ax[idx,0].tick_params(axis='both', which='major', labelsize=fontsize_label)
ax[idx,0].set_xlabel("(f1)",fontsize=fontsize)

ax_inset1 = inset_axes(ax[idx,0], width="20%", height="70%", loc='upper left',\
                       bbox_to_anchor=(.11, -.07, 1, 1), bbox_transform=ax[idx,0].transAxes)

violin_parts = ax_inset1.violinplot(mu, showmeans=False, showmedians=True, widths=.1)
violin_parts['cmedians'].set_color('purple')
ax_inset1.set_xticks([])#[1], [r'$\mu$'])
ax_inset1.set_title(r'$\mu$', fontsize=fontsize)
# ax_inset1.set_yticks([50,60,70,74,80])
ax_inset1.tick_params(axis='both', which='major', labelsize=fontsize_label)


ax_inset2 = inset_axes(ax[idx,0], width="20%", height="70%", loc='upper left',\
                       bbox_to_anchor=(.43, -.07, 1, 1), bbox_transform=ax[idx,0].transAxes)

violin_parts = ax_inset2.violinplot(sigma, showmeans=False, showmedians=True, widths=.1)
violin_parts['cmedians'].set_color('purple')
ax_inset2.set_xticks([])#[1], [r'$\sigma$'])
ax_inset2.set_title(r'$\sigma$', fontsize=fontsize)
# ax_inset2.set_yticks([1,10,20,30])
ax_inset2.tick_params(axis='both', which='major', labelsize=fontsize_label)
# ax_inset2.legend(fontsize=5)



ax[idx,1].fill_between(np.arange(0, 400), np.min(beta, axis=1), np.max(beta, axis=1),
        alpha=0.2, facecolor='#089FFF', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax[idx,1].plot(np.median(beta,axis=1), label=r'Median $R_0(t)$')
ax[idx,1].axvline(347, color='gray', linestyle='dashed')#, label='axvline')
ax[idx,1].axhline(y=1, color='gray', linestyle='dashed')
ax[idx,1].set_xticks([0,200,347,400])
ax[idx,1].set_yticks([1,3,5,7,9])
ax[idx,1].tick_params(axis='both', which='major', labelsize=fontsize_label)
ax[idx,1].set_xlabel("(f2)",fontsize=fontsize)
ax[idx,1].set_ylim(0,12)
ax[idx,1].legend(fontsize=fontsize)


data = pd.read_csv(f'../../data/covid_Republic of Korea.csv', sep='\t')
data['date'] = pd.to_datetime(data['date'])#['I'][640:640+400].to_numpy().reshape([1,-1,1])   
population = 1#data['population'][0]
data_ = get_train_data(data, start=640, length=400, recovery_time=14, estimate=True, prop=True)

ax[idx,2].plot(data_[0,:,0]/population, c='r', label='Estimated j')
ax[idx,2].fill_between(np.arange(0, 400), np.min(pred[:,:,1], axis=0)/population, np.max(pred[:,:,1], axis=0)/population,
        alpha=0.2, facecolor='#089FFF', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax[idx,2].plot(np.median(pred[:,:,1],axis=0)/population, label=r'Median j')
ax[idx,2].axvline(347, color='gray', linestyle='dashed')#, label='axvline')
ax[idx,2].set_xticks([0,200,347,400])
# ax[idx,2].set_yticks([2,4,6,8])
ax[idx,2].tick_params(axis='both', which='major', labelsize=fontsize_label)
ax[idx,2].set_xlabel("(f3)",fontsize=fontsize)
ax[idx,2].set_ylim(0, ax[idx,2].axes.get_ylim()[1]*1.)
ax[idx,2].legend(fontsize=fontsize)
# ax[idx,2].legend(loc='upper right', fontsize=13)


R_e = pred[:,:,0].T*beta/population
ax[idx,1].fill_between(np.arange(0, 400), np.min(R_e, axis=1), np.max(R_e, axis=1),
        alpha=0.2, facecolor='#F39C12', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax[idx,1].plot(np.median(R_e, axis=1), label=r'Median $R_e$')
ax[idx,1].legend(fontsize=fontsize)




pad = 5
# rows = ['Mexico\n(estimated cases)', 'South Africa\n(estimated cases)', 'South Korea\n(estimated cases)']
rows = ['Mexico', 'South Africa', 'South Korea']
for ax_, row in zip(ax[:,0], rows):
    ax_.annotate(row, xy=(0, 0.5), xytext=(-ax_.yaxis.labelpad - pad, 0),
                xycoords=ax_.yaxis.label, textcoords='offset points',
                fontsize=25, ha='center', va='center', rotation=90)
    

cols = ['$K(\cdot)$', '$R_0(t)$', 'j']
for ax_, col in zip(ax[0], cols):
    ax_.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                fontsize=25, ha='center', va='baseline')

# fig.tight_layout()
fig.savefig('./figures/samples_actual_estimate.png', bbox_inches='tight', dpi=300)