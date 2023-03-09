#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 09:45:58 2022

@author: do0236li
"""


import numpy as np
import matplotlib.pyplot as plt


import scipy.stats as stats 
##### without interpolation #####
def f_SIR(y, t, l=1, beta=1.5, gamma=1):
    pre = y[-l:,1]
    integro = sum(pre*dist[-l:,0]*dt)
    
    S, I, R = y[-1]
    
    dSdt = -beta * S * I + integro
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I - integro
    return np.array([[dSdt, dIdt, dRdt]])


length = 1000
t = np.linspace(0., 50., length)
dt = t[1]-t[0]

dist = stats.gamma.pdf(t, a=2, scale=1.2)
dist = dist[::-1].reshape(-1,1)
# dist = dist*dt

# plt.figure()
# plt.plot(dist)


beta, gamma = 1.5, 1
batch = 10
SIR_batch = np.zeros([batch, length, 3])
for j in range(batch):
    SIR_f = np.zeros([length,3])
    S0 = np.random.rand()*.2+.8  # S0 in [0.8, 1]
    I0 = np.random.rand()*.05    # I0 in [ 0 , 0.05]
    R0 = 1-S0-I0
    SIR_f[0,:] = np.array([[1, 0.001, 0]])
    # SIR_f[0,:] = np.array([[S0, I0, R0]])
    
    for i in range(length-1):
        SIR_f[[i+1]] = SIR_f[[i]] + f_SIR(SIR_f[:i+1,:], t, i+1, beta, gamma)*dt
    
    # SIR_batch.append(SIR_f)
    SIR_batch[j,...] = SIR_f
# SIR_batch = np.array(SIR_batch)

# plt.figure()
# plt.plot(SIR_batch[0], label=['s','i','r'])
# plt.legend()

# # np.save('./data/train_sir_l.npy', SIR_batch)
# # np.save('./data/dist_l.npy', dist)








##### with interpolation #####
from scipy.interpolate import interp1d, UnivariateSpline, Rbf
from functools import partial
import scipy.stats as stats 
import numpy as np 
import matplotlib.pyplot as plt

# def integrate_real(pre, t, t_, K):
#     interp_y = interp1d(t, pre, kind='slinear')

#     new_dt = 0.005
#     new_t = np.arange(0, t[-1], new_dt)
#     new_y = interp_y(new_t)
#     new_gamma = K(new_t)[::-1]
#     return sum(new_y*new_gamma)*new_dt

def integrate_real(pre, t, t_tau, K):
    interp_y = interp1d(t_tau, pre, kind='slinear')

    new_dt = 0.005
    num = int(t_tau[-1]//new_dt)
    new_t = np.linspace(0, t_tau[-1], num)
    new_y = interp_y(new_t)
    # print(new_y.shape)
    
    # tt = np.linspace(0,t_tau[-1],num)
    tt = np.linspace(0,t[-1],num)
    # K = partial(Norm, loc=5, scale=1)
    new_gamma = K(tt)[::-1]
    return sum(new_y*new_gamma)*new_dt
    # return sum(new_y*new_gamma)*(tt[1]-tt[0])


def integrate(pre, t, dist):
    interp_y = interp1d(t, pre, kind='slinear')

    new_dt = 0.005
    new_t = np.arange(0, t[-1], new_dt)
    new_y = interp_y(new_t)
    new_gamma = interp_gamma(new_t)[::-1]
    return sum(new_y*new_gamma)*new_dt


tau = 1
def f_SIR(y, t, dt, K, beta=1.5, gamma=1):
    pre = y[:,1]
    
    pre = np.r_[pre, pre[-1]]
    t = np.r_[t,t[-1]+dt]
    # integro = integrate(pre, t, dist)
    integro = integrate_real(pre, t, t*tau, K)
    # print(integro)
    S, I, R = y[-1]
    
    # dSdt = -beta * S * I + integro
    # dIdt = beta * S * I - gamma * I
    # dRdt = gamma * I - integro
    
    dSdt = (-beta * S * I)*tau + integro
    dIdt = (beta * S * I - gamma * I)*tau
    dRdt = (gamma * I)*tau - integro
    
    return np.array([[dSdt, dIdt, dRdt]]) 



def Norm(t, loc, scale):
    return stats.norm.pdf(t, loc=loc, scale=scale)

def Gamma(t, a, scale):
    return stats.gamma.pdf(t, a=a, scale=scale)

Erlang = False

if Erlang==True:
    ##### Erlang distribution #####
    length = 100
    end = 25
    t = np.linspace(0., end, length)
    dt = t[1]-t[0]
    K = partial(Gamma, a=2, scale=1.2)
    
    beta, gamma = 2, 1

else:
    ##### Gaussian distribution #####
    length = 100
    end = 25
    t = np.linspace(0., end, length)
    dt = t[1]-t[0]
    K = partial(Norm, loc=5, scale=1)

    beta, gamma = 2.3, 1

t_fix = np.linspace(0., end, length)
dist = K(t_fix)[::-1]
interp_gamma = interp1d(t_fix, dist, kind='slinear')

# plt.figure()
# plt.plot(dist)



def normal_dist(x, mu=5, sigma=1):
    return 1/(sigma*(2*np.pi)**.5)*np.exp(-1/2*(x-mu)**2/sigma**2)
fig, ax = plt.subplots(1,2)
ax[0].plot(np.linspace(0,10,100), normal_dist(np.linspace(0,10,100), sigma=1, mu=5))
ax[1].plot(np.linspace(0,10*tau,100), normal_dist(np.linspace(0,10*tau,100), sigma=1, mu=5*tau))


batch = 1
SIR_batch = np.zeros([batch, length, 3])
for j in range(batch):
    SIR_f = np.zeros([length,3])
    # S0 = np.random.rand()*.2+.8  # S0 in [0.8, 1]
    # I0 = np.random.rand()*.05    # I0 in [ 0 , 0.05]
    # R0 = 1-S0-I0
    S0, I0, R0 = .99, 0.01, 0
    SIR_f[0,:] = np.array([[S0, I0, R0]])
    
    for i in range(SIR_f.shape[0]-1):
        SIR_f[[i+1]] = SIR_f[[i]] + f_SIR(SIR_f[:i+1,:], t[:i+1], dt, K, beta, gamma)*dt
    
    SIR_batch[j,...] = SIR_f

plt.figure()
plt.plot(SIR_batch[0], label=['s','i','r'])
plt.legend()

# if Erlang==True:
#     np.save('../data/train_sir_l.npy', SIR_batch)
#     np.save('../data/dist_l.npy', dist)
# else:
#     np.save('../data/train_sir_l_norm.npy', SIR_batch)
#     np.save('../data/dist_l_norm.npy', dist)
    
    
    
    
    
    
# fig, ax = plt.subplots(2,2,figsize=(10,5))
# ax = ax.flatten()

# SIR_Erlang = np.load('./data/train_sir_l.npy')
# ax[0].plot(SIR_Erlang[0], label=['s','i','r'])
# ax[0].legend()

# Erlang = np.load('./data/dist_l.npy')
# ax[1].plot(Erlang, label='Erlang')
# ax[1].legend()

# SIR_Gaussian = np.load('./data/train_sir_l_norm.npy')
# ax[2].plot(SIR_Gaussian[0], label=['s','i','r'])
# ax[2].legend()

# Gaussian = np.load('./data/dist_l_norm.npy')
# ax[3].plot(Gaussian, label='Gaussian')
# ax[3].legend()













