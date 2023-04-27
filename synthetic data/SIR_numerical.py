#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 09:14:22 2023

@author: dliu
"""


import numpy as np
import matplotlib.pyplot as plt


from scipy.interpolate import interp1d, UnivariateSpline, Rbf
from functools import partial
import scipy.stats as stats 
import numpy as np 
import matplotlib.pyplot as plt




def integrate_real(pre, t, K):
    interp_y = interp1d(t, pre, kind='slinear')

    dt_new = 0.005
    num = int(t[-1]/dt_new)+1
    t_new = np.linspace(0, t[-1], num)
    y_new = interp_y(t_new)

    K_new = K(t_new)[::-1]
    ## normalize
    K_new = K_new/(K_new.sum()*dt_new)
    
    return sum(y_new*K_new)*dt_new


def f_SIR(y, t, dt, K, beta=1.5, gamma=1, tau=1):
    I_pre = y[:,1]
    I_pre = np.r_[I_pre, I_pre[-1]]
    t = np.r_[t, t[-1]+dt]
    
    integro = integrate_real(I_pre, t, K) 
    S, I, R = y[-1]
    
    dSdt = -beta * S * I + integro
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I - integro
    
    return np.array([[dSdt, dIdt, dRdt]]) * tau


def Norm(t, loc, scale):
    return stats.norm.pdf(t, loc=loc, scale=scale)




##### scale
tau = 1
##### time span
start, end, length = 0, 25, 400
t = np.linspace(start, end, length)
dt = t[1]-t[0]

##### parameters
beta = np.ones([length]) * 2.3
# beta = 1.5 * (1+0.45*np.cos(np.pi*(t+11)/45))
beta[100:] = 1.5
gamma = np.ones([length]) * 1

##### Gaussian distribution #####
mu, sigma = 5, 1
K = partial(Norm, loc=mu, scale=sigma)
dist = K(t)[::-1]

plt.figure()
fig, ax = plt.subplots(1,2,figsize=[10,3])
ax[0].plot(beta, label='beta')
ax[0].legend()
ax[1].plot(dist, label='distribution')
ax[1].legend()

##### Euler method
batch = 1
SIR_batch = np.zeros([batch, length, 4])
for j in range(batch):
    SIR_f = np.zeros([length,3])
    # S0 = np.random.rand()*.2+.8  # S0 in [0.8, 1]
    # I0 = np.random.rand()*.05    # I0 in [ 0 , 0.05]
    # R0 = 1-S0-I0
    S0, I0, R0 = .99, 0.01, 0
    SIR_f[0,:] = np.array([[S0, I0, R0]])
    
    for i in range(SIR_f.shape[0]-1):
        SIR_f[[i+1]] = SIR_f[[i]] + f_SIR(SIR_f[:i+1,:], t[:i+1]*tau, dt, K, beta[i], gamma[i], tau)*dt
    
    SIR_f = np.c_[t.reshape(-1,1)*tau,SIR_f]
    SIR_batch[j,...] = SIR_f

plt.figure()
plt.plot(SIR_f[:,0], SIR_f[:,1:], label=['s','i','r'])
plt.legend()


np.savez('../data/numerical.npz', SIR=SIR_batch, beta=beta, dist=dist, mu=mu, sigma=sigma)