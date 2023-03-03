#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 09:45:58 2022

@author: do0236li
"""


import numpy as np
import matplotlib.pyplot as plt


# ### The SIR model differential equations. ###
# def f_SIR(y, t, beta=.2, gamma=.1):
#     S, I, R = y
    
#     dSdt = -beta * S * I
#     dIdt = beta * S * I - gamma * I
#     dRdt = gamma * I
#     return dSdt, dIdt, dRdt

# from scipy.integrate import odeint as sc_odeint
# t = np.linspace(0, 160, 160)
# # Initial conditions vector
# y0 = 1, 0.001, 0 #s,i,r
# beta, gamma = .5, .1

# # Integrate the SIR equations over the time grid, t.
# ret = sc_odeint(f_SIR, y0, t, args=(beta,gamma))
# plt.plot(ret)



beta = 1.5
gamma = 1

def f_SIR(y, t, beta=.5, gamma=.1):
    S, I, R = y[-1]
    
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return np.array([[dSdt, dIdt, dRdt]])

length = 200
t = np.linspace(0, 80, 200)
dt = t[1]-t[0]
y0 = np.array([[1, 0.001, 0]])

SIR_data = np.zeros([length,3])
SIR_data[[0],:] = y0

for i in range(1,length):
    SIR_data[[i]] = SIR_data[[i-1]] + f_SIR(SIR_data[[i-1]], t, beta, gamma)*dt

plt.figure()
plt.plot(SIR_data, label=['s','i','r'])
plt.legend()





import scipy.stats as stats 
k = 1
t = np.linspace(0., 80.//k, 200//k)

dist = stats.gamma.pdf(t, a=2, scale=1.2)
dist = dist[::-1].reshape(-1,1)

dx = 1/dist.sum()
dist = dist*dx
np.save('./data/dist.npy', dist)

plt.figure()
plt.plot(dist)


def f_SIR(y, t, l=1, beta=1.5, gamma=1):
    pre = y[-l:,1]
    S, I, R = y[-1]
    
    
    dSdt = -beta * S * I + sum(pre*dist[-l:,0])
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I - sum(pre*dist[-l:,0])
    return np.array([[dSdt, dIdt, dRdt]])


batch = 100
SIR_batch = []
for _ in range(batch):
    length = 40
    SIR_f = np.zeros([length,3])
    S0 = np.random.rand()*.1+.9  # S0 in [0.8, 1]
    I0 = np.random.rand()*.05    # I0 in [ 0 , 0.05]
    R0 = 1-S0-I0
    # SIR_f[0,:] = np.array([[1, 0.001, 0]])
    SIR_f[0,:] = np.array([[S0, I0, R0]])
    
    for i in range(length-1):
        SIR_f[[i+1]] = SIR_f[[i]] + f_SIR(SIR_f[:i+1,:], t, i+1, beta, gamma)*dt
    
    SIR_batch.append(SIR_f)

SIR_batch = np.array(SIR_batch)

plt.figure()
plt.plot(SIR_batch[0], label=['s','i','r'])
plt.legend()

np.save('./data/train_sir.npy', SIR_batch)






batch = 100
SIR_batch = []
for _ in range(batch):
    length = 40
    SIR_f = np.zeros([length,3])
    S0 = np.random.rand()*.2+.8  # S0 in [0.8, 1]
    I0 = np.random.rand()*.05    # I0 in [ 0 , 0.05]
    R0 = 1-S0-I0
    # SIR_f[0,:] = np.array([[1, 0.001, 0]])
    SIR_f[0,:] = np.array([[S0, I0, R0]])
    
    for i in range(length-1):
        SIR_f[[i+1]] = SIR_f[[i]] + f_SIR(SIR_f[:i+1,:], t, i+1, beta, gamma)*dt
    
    SIR_batch.append(SIR_f)

SIR_batch = np.array(SIR_batch)

plt.figure()
plt.plot(SIR_batch[0], label=['s','i','r'])
plt.legend()

np.save('./data/test_sir.npy', SIR_batch)







# import matplotlib.pyplot as plt
# from torchdiffeq import odeint
# import numpy as np

# # from _impl import odeint
# import torch
# import torch.nn as nn
# import torch.optim as optim

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# data = np.load('sir.npy')

# beta = 1.5 #.5
# gamma = 1 #.1

# class ODEFunc(nn.Module):

#     def __init__(self, time):
#         super(ODEFunc, self).__init__()

#         self.memory = nn.Sequential(
#             nn.Linear(1, 20),
#             nn.Tanh(),
#             nn.Linear(20, 20),
#             nn.Tanh(),
#             nn.Linear(20, 20),
#             nn.Tanh(),
#             nn.Linear(20, 1),
#         )

#         for m in self.memory.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, mean=0, std=0.1)
#                 nn.init.constant_(m.bias, val=0)
                
#     def forward(self, t, y, y_series=None, time=None, j=None):
#         S, I, R = torch.split(y,1,dim=1)
#         # print(I.shape)
        
#         dSdt = -beta * S * I #+ self.integro# + self.memory(I)#sum(pre*dist)*dx
#         dIdt = beta * S * I - gamma * I
#         dRdt = gamma * I #- self.integro#- self.memory(I)#sum(pre*dist)*dx
#         return torch.cat((dSdt,dIdt,dRdt),1)
#         # return self.net(y**3)
        

# k = 5
# t = torch.linspace(0., 80./k, 200//k).to(device)


# func = ODEFunc(t).to(device)
# y = torch.tensor(data, dtype=torch.float32).to(device)  
# y0 = y[[0,1],:].to(device)

# pred_y = odeint(func, y0, t, method='euler').to(device)

# plt.figure()
# plt.plot(pred_y[:,0,:].cpu().detach(), label=['S', 'I', 'R'])
# plt.legend()