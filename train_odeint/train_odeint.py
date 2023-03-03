#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:06:35 2022

@author: dliu
"""


import matplotlib.pyplot as plt
# from torchdiffeq import odeint
import numpy as np

from _impl import odeint
# from _impl_origin import odeint

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data = np.load('../data/train_sir.npy')
# plt.plot(data)


import scipy.stats as stats
l = 30
# x = np.linspace (0, 8, l)
t = torch.linspace(0., 80., 200)

dist = stats.gamma.pdf(t, a=2, scale=1)
# plt.plot(dist)
dist = dist[::-1]

dx = 1/dist.sum()



beta = 1.5 #.5
gamma = 1 #.1


class Memory(nn.Module):    
    def __init__(self):
        super(Memory, self).__init__()

        self.memory = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1),
        )
    def forward(self, t):
        
        t = torch.flip(t, [0]).reshape([-1,1])
        t = t.repeat(1,2).reshape(-1,1)
        return self.memory(t)

class ODEFunc(nn.Module):

    def __init__(self, time):
        super(ODEFunc, self).__init__()

        self.memory = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1),
        )

        for m in self.memory.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
                
        self.mm = None
        self.time = time
        
        self.g = Memory().to(device)#(time)
        # print('d', self.g.shape)
        
    def forward(self, t, y, y_series=None, time=None, j=None):
        # if y_series is not None:
        #     self.integro = self._memory(y_series, self.time, j)
        if t.item()==0:
            self.gg = self.g(time)
        self.integro = self.integral(y_series,self.gg,j)
        print('ina', self.integro.shape)
        # self.integro = self.integral(y_series, self.g, j)
        
        # print(self.integro.shape)
        # pre = y[-l:,1]
        S, I, R = torch.split(y,1,dim=1)
        # print(I.shape)
        
        dSdt = -beta * S * I + self.integro# + self.memory(I)#sum(pre*dist)*dx
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I - self.integro#- self.memory(I)#sum(pre*dist)*dx
        return torch.cat((dSdt,dIdt,dRdt),1)
        # return self.net(y**3)
    
    def _memory(self, solution, time_grid, j):
        import torch
        S, I, R = torch.split(solution,1,dim=2)
        
        dt = 0.4103
        batchsize = S.shape[1]
        # print(batchsize)
        
        t = torch.flip(time_grid, [0]).reshape([-1,1])
        # print(t)
        gamma = self.memory(t)[-j:]
        gamma = gamma.repeat(1,batchsize)
        # print(gamma.shape)
        # print(I[:,:,0].shape)
        # return func.memory(torch.tensor([[1.]]).cuda())
        integro = I[:,:,0] * gamma * dt
        # print('integrao', integro.sum(0).reshape(batchsize,1))
        return integro.sum(0).reshape(batchsize,1)
    
    def _out(self, time):
        t = torch.flip(time, [0]).reshape([-1,1])
        # print(t)
        batchsize = 2
        gamma = self.memory(t).repeat(1,batchsize)
        return gamma
    
    def integral(self, solution, g, j):
        g = g.reshape(40,2)

        S, I, R = torch.split(solution,1,dim=2)

        dt = 0.4103
        batchsize = S.shape[1]
        
        gamma = g[-j:]
        # gamma = gamma.repeat(1,batchsize)

        integro = I[:,:,0] * gamma * dt
        # print('integrao', integro.sum(0).reshape(batchsize,1))
        return integro.sum(0).reshape(batchsize,1)
    
if __name__ == '__main__':

    k = 5
    t = torch.linspace(0., 80./k, 200//k).to(device)
    
    
    func = ODEFunc(t).to(device)
    y = torch.tensor(data, dtype=torch.float32).to(device)  
    y0 = y[[0,1],:].to(device)
    
    y0 = y[[0,1],0,:].to(device)
    pred_y = odeint(func, y0, t, method='euler').to(device)
    plt.plot(pred_y[:,0,:].cpu().detach(), label=['S', 'I', 'R'])
    plt.legend()
    
    # optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    
    # batch_size = 2
    # batch = data[np.arange(batch_size).reshape(-1,1)+np.arange(200//5)]
    # batch_y = torch.tensor(batch, dtype=torch.float32).to(device)
    # batch_t = t
    
    # batch_y0 = batch_y[:,0,:].to(device)
    # for itr in range(1, 1000):
    #     optimizer.zero_grad()
    #     pred_y = odeint(func, batch_y0, batch_t, method='euler').to(device)
    #     pred_y = pred_y.transpose(1,0)
    #     loss = torch.mean(torch.abs(pred_y - batch_y))
    #     loss.backward()
    #     optimizer.step()
        
    #     print(loss.item())