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
        'size'   : 14}
matplotlib.rc('font', **font)

simulation = np.load('../../data/simulation_2_3.npy')[:400]

path = glob.glob('samples_/simulation*')
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
    distribution.append(dist[::-1])
    # plt.plot(t,dist[::-1])

distribution=np.array(distribution).T
# plt.plot(t, distribution)






mu_3 = mu[:20]
sigma_3 = sigma[:20]
beta_3 = beta[:,:20]
distribution_3 = distribution[:,:20]
pred_3 = pred[:20]

mu_6 = mu[20:40]
sigma_6 = sigma[20:40]
beta_6 = beta[:,20:40]
distribution_6 = distribution[:,20:40]
pred_6 = pred[20:40]

mu_10 = mu[40:60]
sigma_10 = sigma[40:60]
beta_10 = beta[:,40:60]
distribution_10 = distribution[:,40:60]
pred_10 = pred[40:60]


### by tanh as last layer
# mu_6 = mu[60:80]
# sigma_6 = sigma[60:80]
# beta_6 = beta[:,60:80]
# distribution_6 = distribution[:,60:80]
# pred_6 = pred[60:80]



mu_real = 70
sigma_real = 1
beta_real = 2.66
beta_simulation = 2.3*8/9

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig, ax = plt.subplots(3,3, figsize=[18,12])#, gridspec_kw={'width_ratios': [1,1,1], 'height_ratios': [1.5,1.5,1,1,1]})
fontsize = 16

ax_K3 = ax[0,0]
ax_K3.plot(t,distribution_3, alpha=.5)
mu_median = round(np.median(mu_3))
ax_K3.axvline(347-mu_median, color='purple', label='axvline')
ax_K3.set_xticks([0,347-200,347-mu_median,347], [347,200,mu_median,0])
# ax_K3.set_yticks([])
ax_K3.tick_params(axis='both', which='major', labelsize=11)
ax_K3.set_xlabel("(a1)",fontsize=fontsize)

ax_inset1 = inset_axes(ax_K3, width="20%", height="70%", loc='upper left',\
                       bbox_to_anchor=(.1, -.07, 1, 1), bbox_transform=ax_K3.transAxes)

violin_parts = ax_inset1.violinplot(mu_3, showmeans=False, showmedians=True, widths=.1)
violin_parts['cmedians'].set_color('purple')
ax_inset1.set_xticks([])#[1], [r'$\mu$'])
ax_inset1.set_title(r'$\mu$', fontsize=fontsize)
ax_inset1.axhline(y=mu_real, color='r', linestyle='dashed', label='axvline')
ax_inset1.set_yticks([50,mu_median,70])
ax_inset1.tick_params(axis='both', which='major', labelsize=11)


ax_inset2 = inset_axes(ax_K3, width="20%", height="70%", loc='upper left',\
                       bbox_to_anchor=(.42, -.07, 1, 1), bbox_transform=ax_K3.transAxes)

violin_parts = ax_inset2.violinplot(sigma_3, showmeans=False, showmedians=True, widths=.1)
violin_parts['cmedians'].set_color('purple')
ax_inset2.set_xticks([])#[1], [r'$\sigma$'])
ax_inset2.set_title(r'$\sigma$', fontsize=fontsize)
ax_inset2.axhline(y=sigma_real, color='r', linestyle='dashed', label='real $\sigma$')
ax_inset2.set_yticks([1,10,20,30])
ax_inset2.tick_params(axis='both', which='major')
# ax_inset2.legend(fontsize=5)


ax_R3 = ax[1,0]
# ax_R3.axhline(y=beta_real, color='r', linestyle='dashed', label=r'$R_0$')
ax_R3.plot(simulation[:,0]*beta_simulation, c='r',  linestyle='dashed', label=r'Synthetic $R_e$')
ax_R3.axhline(y=1, color='gray', linestyle='dashed')
ax_R3.fill_between(np.arange(0, 400), np.min(beta_3, axis=1), np.max(beta_3, axis=1),
        alpha=0.2, facecolor='#089FFF', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax_R3.plot(np.median(beta_3,axis=1), label=r'Median $R_0(t)$')

R_e3 = pred_3[:,:,0].T*beta_3
ax_R3.fill_between(np.arange(0, 400), np.min(R_e3, axis=1), np.max(R_e3, axis=1),
        alpha=0.2, facecolor='#F39C12', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax_R3.plot(np.median(R_e3, axis=1), label=r'Median $R_e$')


ax_R3.axvline(347, color='gray', linestyle='dashed')#, label='axvline')
ax_R3.set_xticks([0,200,347,400])
# ax_R3.set_yticks([1,beta_simulation,4,6,8])
ax_R3.set_yticks([1,3,5,7,9])
ax_R3.set_ylim(0, 10)
ax_R3.tick_params(axis='both', which='major')
ax_R3.set_xlabel("(a2)",fontsize=fontsize)
ax_R3.legend(fontsize=fontsize)


ax_I3 = ax[2,0]
ax_I3.plot(simulation[:,1], c='r', label='Synthetic j')
ax_I3.fill_between(np.arange(0, 400), np.min(pred_3[:,:,1], axis=0), np.max(pred_3[:,:,1], axis=0),
        alpha=0.2, facecolor='#089FFF', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax_I3.plot(np.median(pred_3[:,:,1],axis=0), label=r'Median j')
ax_I3.axvline(347, color='gray', linestyle='dashed')#, label='axvline')
ax_I3.set_xticks([0,200,347,400])
ax_I3.set_xlabel("(a3)",fontsize=fontsize)
ax_I3.legend(fontsize=fontsize)

# ax_S3 = ax[3,0]
# ax_S3.plot(simulation[:,0], c='r', label='Synthetic S')
# ax_S3.fill_between(np.arange(0, 400), np.min(pred_3[:,:,0], axis=0), np.max(pred_3[:,:,0], axis=0),
#         alpha=0.2, facecolor='#089FFF', #edgecolor='#F39C12', linewidth=1, 
#         linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
# ax_S3.plot(np.median(pred_3[:,:,0],axis=0), label=r'median S')
# ax_S3.axvline(347, color='gray', linestyle='dashed')#, label='axvline')
# ax_S3.set_xticks([0,200,347,400])
# ax_S3.set_xlabel("(a4)",fontsize=fontsize)
# ax_S3.legend(fontsize=fontsize)

# ax_r3 = ax[4,0]
# ax_r3.plot(simulation[:,2], c='r', label='Synthetic R')
# ax_r3.fill_between(np.arange(0, 400), np.min(pred_3[:,:,2], axis=0), np.max(pred_3[:,:,2], axis=0),
#         alpha=0.2, facecolor='#089FFF', #edgecolor='#F39C12', linewidth=1, 
#         linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
# ax_r3.plot(np.median(pred_3[:,:,2],axis=0), label=r'median R')
# ax_r3.axvline(347, color='gray', linestyle='dashed')#, label='axvline')
# ax_r3.set_xticks([0,200,347,400])
# ax_r3.set_ylim(0,1.1)
# ax_r3.set_xlabel("(a5)",fontsize=fontsize)
# ax_r3.legend(fontsize=fontsize)





ax_K6 = ax[0,1]
ax_K6.plot(t,distribution_6, alpha=.5)
mu_median = round(np.median(mu_6))
ax_K6.axvline(347-mu_median, color='purple', label='axvline')
ax_K6.set_xticks([0,347-200,347-mu_median,347], [347,200,mu_median,0])
# ax_K6.set_yticks([])
ax_K6.tick_params(axis='both', which='major', labelsize=11)
ax_K6.set_xlabel("(b1)",fontsize=fontsize)

ax_inset1 = inset_axes(ax_K6, width="20%", height="70%", loc='upper left',\
                       bbox_to_anchor=(.1, -.07, 1, 1), bbox_transform=ax_K6.transAxes)

violin_parts = ax_inset1.violinplot(mu_6, showmeans=False, showmedians=True, widths=.1)
violin_parts['cmedians'].set_color('purple')
ax_inset1.set_xticks([])#[1], [r'$\mu$'])
ax_inset1.set_title(r'$\mu$', fontsize=fontsize)
ax_inset1.axhline(y=mu_real, color='r', linestyle='dashed', label='axvline')
ax_inset1.set_yticks([60, mu_median, 70, 80])
ax_inset1.tick_params(axis='both', which='major')


ax_inset2 = inset_axes(ax_K6, width="20%", height="70%", loc='upper left',\
                       bbox_to_anchor=(.42, -.07, 1, 1), bbox_transform=ax_K6.transAxes)

violin_parts = ax_inset2.violinplot(sigma_6, showmeans=False, showmedians=True, widths=.1)
violin_parts['cmedians'].set_color('purple')
ax_inset2.set_xticks([])#[1], [r'$\sigma$'])
ax_inset2.set_title(r'$\sigma$', fontsize=fontsize)
ax_inset2.axhline(y=sigma_real, color='r', linestyle='dashed', label='real $\sigma$')
ax_inset2.set_yticks([1,10,20,30])
ax_inset2.tick_params(axis='both', which='major')
# ax_inset2.legend(fontsize=5)


ax_R6 = ax[1,1]
# ax_R6.axhline(y=beta_real, color='r', linestyle='dashed', label=r'$R_0$')
ax_R6.plot(simulation[:,0]*beta_simulation, c='r',  linestyle='dashed', label=r'Synthetic $R_e$')
ax_R6.axhline(y=1, color='gray', linestyle='dashed')
ax_R6.fill_between(np.arange(0, 400), np.min(beta_6, axis=1), np.max(beta_6, axis=1),
        alpha=0.2, facecolor='#089FFF', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax_R6.plot(np.median(beta_6,axis=1), label=r'Median $R_0(t)$')

R_e6 = pred_6[:,:,0].T*beta_6
ax_R6.fill_between(np.arange(0, 400), np.min(R_e6, axis=1), np.max(R_e6, axis=1),
        alpha=0.2, facecolor='#F39C12', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax_R6.plot(np.median(R_e6, axis=1), label=r'Median $R_e$')


ax_R6.axvline(347, color='gray', linestyle='dashed')#, label='axvline')
ax_R6.set_xticks([0,200,347,400])
# ax_R6.set_yticks([1,beta_simulation,4,6,8])
ax_R6.set_yticks([1,3,5,7,9])
ax_R6.set_ylim(0, 10)
ax_R6.tick_params(axis='both', which='major', labelsize=11)
ax_R6.set_xlabel("(b2)",fontsize=fontsize)
ax_R6.legend(fontsize=fontsize)


ax_I6 = ax[2,1]
ax_I6.plot(simulation[:,1], c='r', label=r'Synthetic j')
ax_I6.fill_between(np.arange(0, 400), np.min(pred_6[:,:,1], axis=0), np.max(pred_6[:,:,1], axis=0),
        alpha=0.2, facecolor='#089FFF', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax_I6.plot(np.median(pred_6[:,:,1],axis=0), label=r'Median j')
ax_I6.axvline(347, color='gray', linestyle='dashed')#, label='axvline')
ax_I6.set_xticks([0,200,347,400])
ax_I6.set_xlabel("(b3)",fontsize=fontsize)
ax_I6.legend(fontsize=fontsize)

# ax_S6 = ax[3,1]
# ax_S6.plot(simulation[:,0], c='r', label=r'Synthetic S')
# ax_S6.fill_between(np.arange(0, 400), np.min(pred_6[:,:,0], axis=0), np.max(pred_6[:,:,0], axis=0),
#         alpha=0.2, facecolor='#089FFF', #edgecolor='#F39C12', linewidth=1, 
#         linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
# ax_S6.plot(np.median(pred_6[:,:,0],axis=0), label=r'median S')
# ax_S6.axvline(347, color='gray', linestyle='dashed')#, label='axvline')
# ax_S6.set_xticks([0,200,347,400])
# ax_S6.set_xlabel("(b4)",fontsize=fontsize)
# ax_S6.legend(fontsize=fontsize)

# ax_r6 = ax[4,1]
# ax_r6.plot(simulation[:,2], c='r', label=r'Synthetic R')
# ax_r6.fill_between(np.arange(0, 400), np.min(pred_6[:,:,2], axis=0), np.max(pred_6[:,:,2], axis=0),
#         alpha=0.2, facecolor='#089FFF', #edgecolor='#F39C12', linewidth=1, 
#         linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
# ax_r6.plot(np.median(pred_6[:,:,2],axis=0), label=r'median R')
# ax_r6.axvline(347, color='gray', linestyle='dashed')#, label='axvline')
# ax_r6.set_xticks([0,200,347,400])
# ax_r6.set_ylim(0,1.1)
# ax_r6.set_xlabel("(b5)",fontsize=fontsize)
# ax_r6.legend(fontsize=fontsize)





ax_K10 = ax[0,2]
ax_K10.plot(t,distribution_10, alpha=.5)
mu_median = round(np.median(mu_10))
ax_K10.axvline(347-mu_median, color='purple', label='axvline')
ax_K10.set_xticks([0,347-200,347-mu_median,347], [347,200,mu_median,0])
# ax_K10.set_yticks([])
ax_K10.tick_params(axis='both', which='major', labelsize=11)
ax_K10.set_xlabel("(c1)",fontsize=fontsize)

ax_inset1 = inset_axes(ax_K10, width="20%", height="70%", loc='upper left',\
                       bbox_to_anchor=(.1, -.07, 1, 1), bbox_transform=ax_K10.transAxes)

violin_parts = ax_inset1.violinplot(mu_10, showmeans=False, showmedians=True, widths=.1)
violin_parts['cmedians'].set_color('purple')
ax_inset1.set_xticks([])#[1], [r'$\mu$'])
ax_inset1.set_title(r'$\mu$', fontsize=13)
ax_inset1.axhline(y=mu_real, color='r', linestyle='dashed', label='axvline')
ax_inset1.set_yticks([50,70,mu_median,80])
ax_inset1.tick_params(axis='both', which='major')
# for pc in violin_parts['bodies']:
#     pc.set_facecolor('red')
#     pc.set_edgecolor('black')
# x = np.random.normal(1, 0.01, size=len(mu_10))
# x[mu_10.argmin()]=1
# from matplotlib import cm
# ax_inset1.scatter(x,mu_10, c=cm.prism(3), alpha=0.2, s=20)

ax_inset2 = inset_axes(ax_K10, width="20%", height="70%", loc='upper left',\
                       bbox_to_anchor=(.42, -.07, 1, 1), bbox_transform=ax_K10.transAxes)

violin_parts = ax_inset2.violinplot(sigma_10, showmeans=False, showmedians=True, widths=.1)
violin_parts['cmedians'].set_color('purple')
ax_inset2.set_xticks([])#[1], [r'$\sigma$'])
ax_inset2.set_title(r'$\sigma$', fontsize=13)
ax_inset2.axhline(y=sigma_real, color='r', linestyle='dashed', label='real $\sigma$')
ax_inset2.set_yticks([1,10,20,30])
ax_inset2.tick_params(axis='both', which='major')
# ax_inset2.legend(fontsize=5)


ax_R10 = ax[1,2]
# ax_R10.axhline(y=beta_real, color='r', linestyle='dashed', label=r'$R_0$')
ax_R10.plot(simulation[:,0]*beta_simulation, c='r',  linestyle='dashed', label=r'Synthetic $R_e$')
ax_R10.axhline(y=1, color='gray', linestyle='dashed')
ax_R10.fill_between(np.arange(0, 400), np.min(beta_10, axis=1), np.max(beta_10, axis=1),
        alpha=0.2, facecolor='#089FFF', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax_R10.plot(np.median(beta_10,axis=1), label=r'Median $R_0(t)$')

R_e10 = pred_10[:,:,0].T*beta_10
ax_R10.fill_between(np.arange(0, 400), np.min(R_e10, axis=1), np.max(R_e10, axis=1),
        alpha=0.2, facecolor='#F39C12', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax_R10.plot(np.median(R_e10, axis=1), label=r'Median $R_e$')


ax_R10.axvline(347, color='gray', linestyle='dashed')#, label='axvline')
ax_R10.set_xticks([0,200,347,400])
# ax_R10.set_yticks([1,beta_simulation,4,6,8])
ax_R10.set_yticks([1,3,5,7,9])
ax_R10.set_ylim(0, 10)
ax_R10.tick_params(axis='both', which='major')
ax_R10.set_xlabel("(c2)",fontsize=fontsize)
ax_R10.legend(fontsize=fontsize)



ax_I10 = ax[2,2]
ax_I10.plot(simulation[:,1], c='r', label='Synthetic j')
ax_I10.fill_between(np.arange(0, 400), np.min(pred_10[:,:,1], axis=0), np.max(pred_10[:,:,1], axis=0),
        alpha=0.2, facecolor='#089FFF', #edgecolor='#F39C12', linewidth=1, 
        linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
ax_I10.plot(np.median(pred_10[:,:,1],axis=0), label=r'Median j')
ax_I10.axvline(347, color='gray', linestyle='dashed')#, label='axvline')
ax_I10.set_xticks([0,200,347,400])
ax_I10.set_xlabel("(c3)",fontsize=fontsize)
ax_I10.legend(fontsize=fontsize)

# ax_S10 = ax[3,2]
# ax_S10.plot(simulation[:,0], c='r', label='Synthetic S')
# ax_S10.fill_between(np.arange(0, 400), np.min(pred_10[:,:,0], axis=0), np.max(pred_10[:,:,0], axis=0),
#         alpha=0.2, facecolor='#089FFF', #edgecolor='#F39C12', linewidth=1, 
#         linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
# ax_S10.plot(np.median(pred_10[:,:,0],axis=0), label=r'median S')
# ax_S10.axvline(347, color='gray', linestyle='dashed')#, label='axvline')
# ax_S10.set_xticks([0,200,347,400])
# ax_S10.set_xlabel("(c4)",fontsize=fontsize)
# ax_S10.legend(fontsize=fontsize)

# ax_r10 = ax[4,2]
# ax_r10.plot(simulation[:,2], c='r', label='Synthetic R')
# ax_r10.fill_between(np.arange(0, 400), np.min(pred_10[:,:,2], axis=0), np.max(pred_10[:,:,2], axis=0),
#         alpha=0.2, facecolor='#089FFF', #edgecolor='#F39C12', linewidth=1, 
#         linestyle='dashdot', antialiased=True)#, label=r'samples of $R_0(t)$')
# ax_r10.plot(np.median(pred_10[:,:,2],axis=0), label=r'median R')
# ax_r10.axvline(347, color='gray', linestyle='dashed')#, label='axvline')
# ax_r10.set_xticks([0,200,347,400])
# ax_r10.set_ylim(0,1.1)
# ax_r10.set_xlabel("(c5)",fontsize=fontsize)
# ax_r10.legend(fontsize=fontsize)


pad = 5
rows = ['$K(\cdot)$', '$R_0(t)$', 'j', 's', 'r']
for ax_, row in zip(ax[:,0], rows):
    ax_.annotate(row, xy=(0, 0.5), xytext=(-ax_.yaxis.labelpad - pad, 0),
                xycoords=ax_.yaxis.label, textcoords='offset points',
                fontsize=25, ha='right', va='center', rotation=90)
    
cols = ['$R_0(t)\in(0,3.5)$', '$R_0(t)\in(0,6)$','$R_0(t)\in(0,10)$'] 
for ax_, col in zip(ax[0], cols):
    ax_.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                fontsize=25, ha='center', va='baseline')

# fig.tight_layout(pad=0, w_pad=0, h_pad=0)
fig.savefig('./figures/samples_simulation.png', bbox_inches='tight', dpi=300)