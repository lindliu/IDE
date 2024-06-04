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

# import numpy as np
# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt

# def f_SIR(t, y, beta, gamma):
#     S, I, R = y[0], y[1], y[2]
#     # beta, gamma = 2.5, 1.25
    
#     dSdt = -beta * S * I
#     dIdt = beta * S * I - gamma * I
#     dRdt = gamma * I
    
#     return [dSdt, dIdt, dRdt]

# t = np.linspace(0,10)
# sol = solve_ivp(f_SIR, [0,10], [.99, .01, 0], args=(2.3, 1), dense_output=True)

# z = sol.sol(t)
# plt.plot(t, z.T, label=['s', 'i', 'r'])
# plt.legend()


def integrate_real(pre, t, K):
    interp_y = interp1d(t, pre, kind='slinear')

    dt_new = 0.005
    num = int(t[-1]/dt_new)+1
    t_new = np.linspace(0, t[-1], num)
    y_new = interp_y(t_new)

    K_new = K(t_new)[::-1]
    ## normalize
    K_new = K_new/(K_new.sum()*dt_new+1e-16)
    
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
start, end, length = 0, 400/14, 400
t = np.linspace(start, end, length)
dt = t[1]-t[0]

##### parameters
beta = np.ones([length]) * 2.66#2.3 # 
# beta = 1.5 * (1+0.45*np.cos(np.pi*(t+11)/45))
# beta[100:] = 1.5
gamma = np.ones([length]) * 1

##### Gaussian distribution #####
mu, sigma = 70/14, 1/14
K = partial(Norm, loc=mu, scale=sigma)
dist = K(t)[::-1]

plt.figure()
fig, ax = plt.subplots(1,2,figsize=[10,3])
ax[0].plot(beta, label='beta')
ax[0].legend()
ax[1].plot(dist, label='distribution')
ax[1].legend()


simulation = np.load('../../data/simulation_2_3.npy')

##### Euler method
batch = 1
SIR_batch = np.zeros([batch, length, 4])
for j in range(batch):
    SIR_f = np.zeros([length,3])
    # S0 = np.random.rand()*.2+.8  # S0 in [0.8, 1]
    # I0 = np.random.rand()*.05    # I0 in [ 0 , 0.05]
    # R0 = 1-S0-I0
    
    
    # S0, I0, R0 = .99, 0.01, 0
    S0, I0, R0 = simulation[0]
    
    SIR_f[0,:] = np.array([[S0, I0, R0]])
    
    for i in range(SIR_f.shape[0]-1):
        SIR_f[[i+1]] = SIR_f[[i]] + f_SIR(SIR_f[:i+1,:], t[:i+1]*tau, dt, K, beta[i], gamma[i], tau)*dt
    
    SIR_f = np.c_[t.reshape(-1,1)*tau,SIR_f]
    SIR_batch[j,...] = SIR_f


np.savez('../../data/numerical.npz', SIR=SIR_batch, beta=beta, dist=dist, mu=mu, sigma=sigma)


print(f'numerical beta: {beta[0]}, simulation beta: 2.3')
print(f'MSE between simulations and numerical results: {np.mean((simulation[:400,1]-SIR_f[:,2])**2):.2E}')


# plt.figure()
# fig, ax = plt.subplots(1,4,figsize=[15,3])

# ax[0].plot(SIR_f[:,0]*14, simulation[:400,1], c='r', label='Synthetic I')
# ax[0].plot(SIR_f[:,0]*14, SIR_f[:,2], label='Numerical I')
# ax[0].legend()
# ax[0].set_xlabel('(a)')
# ax[0].tick_params(axis='both', which='major', labelsize=8)

# ax[1].plot(SIR_f[:,0]*14, simulation[:400,0], c='r', label='Synthetic S')
# ax[1].plot(SIR_f[:,0]*14, SIR_f[:,1], label='Numerical S')
# ax[1].legend()
# ax[1].set_ylim(0,1)
# ax[1].set_xlabel('(b)')
# ax[1].tick_params(axis='both', which='major', labelsize=8)

# ax[2].plot(SIR_f[:,0]*14, simulation[:400,2], c='r', label='Synthetic R')
# ax[2].plot(SIR_f[:,0]*14, SIR_f[:,3], label='Numerical R')
# ax[2].legend()
# ax[2].set_ylim(0,1)
# ax[2].set_xlabel('(c)')
# ax[2].tick_params(axis='both', which='major', labelsize=8)


# ax[3].plot(SIR_f[:,0]*14, SIR_f[:400,1]*2.66, c='r', label='real $R_e$')
# ax[3].plot(SIR_f[:,0]*14, simulation[:400,0]*2.66, label='Numerical $R_e$')
# ax[3].legend()
# ax[3].set_ylim(0,5)
# ax[3].set_xlabel('(c)')
# ax[3].tick_params(axis='both', which='major', labelsize=8)



fontsize = 12
plt.figure()
fig, ax = plt.subplots(1,4,figsize=[12,2.3])

ax[0].plot(SIR_f[:,0]*14, simulation[:400,0], c='r', label='Synthetic s')
ax[0].set_ylim(0,1.1)
ax[0].tick_params(axis='both', which='major', labelsize=8)
ax[0].set_xlabel('(a)', fontsize=fontsize)
ax[0].legend(fontsize=fontsize, loc='upper right')

ax[1].plot(SIR_f[:,0]*14, simulation[:400,1], c='r', label='Synthetic j')
ax[1].tick_params(axis='both', which='major', labelsize=8)
ax[1].set_xlabel('(b)', fontsize=fontsize)
ax[1].legend(fontsize=fontsize, loc='upper right')

ax[2].plot(SIR_f[:,0]*14, simulation[:400,2], c='r', label='Synthetic r')
ax[2].set_ylim(0,1.1)
ax[2].tick_params(axis='both', which='major', labelsize=8)
ax[2].set_xlabel('(c)', fontsize=fontsize)
ax[2].legend(fontsize=fontsize, loc='upper right')

ax[3].plot(SIR_f[:,0]*14, simulation[:400,0]*2.3*8/9, c='r', label='Synthetic $R_e$')
ax[3].set_ylim(0,3)
ax[3].tick_params(axis='both', which='major', labelsize=8)
ax[3].set_xlabel('(d)', fontsize=fontsize)
ax[3].legend(fontsize=fontsize, loc='upper right')
# ax[3].set_yticks([1,2,3])

fig.savefig('./figures/synthetic_data.png', bbox_inches='tight', dpi=300)


# plt.figure()
# fig, ax = plt.subplots(1,1,figsize=[4,2])

# ax.plot(SIR_f[:,0]*14, simulation[:400,1], c='r', label='Synthetic j')
# ax.plot(SIR_f[:,0]*14, SIR_f[:,2], label='Numerical j')
# ax.legend()
# ax.set_xlabel('days')
# ax.tick_params(axis='both', which='major', labelsize=8)

# fig.savefig('synthetic_numerical.png', bbox_inches='tight', dpi=300)


