#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:06:35 2022

@author: dliu
"""


import matplotlib.pyplot as plt
# from torchdiffeq import odeint
import numpy as np

# from _impl import odeint
from _impl_origin import odeint

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
            # nn.ReLU()
            nn.Sigmoid()
        )
        
        for m in self.memory.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)
                
        
        self.mu = nn.Parameter(torch.tensor(5.).to(device), requires_grad=True)  ## initial value matters, if we choose 1.5 then it fails
        self.sigma = nn.Parameter(torch.tensor(1.).to(device), requires_grad=True)
        
    # def forward(self, t):
    #     return self.memory(t)
    
    def forward(self, t):
        return 1/(self.sigma*(2*torch.pi)**.5)*torch.exp(-1/2*(t-self.mu)**2/self.sigma**2)


# Erlange = False
# if Erlange==True:    
#     t = torch.linspace(0., 15, 1000).to(device)
#     dist = np.load('../data/dist_l.npy')
# else:
#     t = torch.linspace(0., 15, 100).to(device)
#     dist = np.load('../data/dist_l_norm.npy')

# dist = torch.tensor(dist, dtype=torch.float32).to(device)

# batch_t = t.reshape(-1,1)
# batch_dist = dist.reshape(-1,1)

# func_m = Memory().to(device)
# optimizer = optim.RMSprop(func_m.parameters(), lr=1e-3)

# for itr in range(1, 20000):
#     optimizer.zero_grad()
    
#     pred_dist = func_m(batch_t)
#     pred_dist = torch.flip(pred_dist, dims=(0,))
#     loss = nn.functional.mse_loss(pred_dist, batch_dist)

#     loss.backward()
#     optimizer.step()
    
#     if itr%1000==0:
#         print(loss.item())

# pred_dist = func_m(batch_t)  
# plt.plot(pred_dist.cpu().detach().numpy()[::-1])
# plt.plot(batch_dist.cpu())



class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.NN = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
            # nn.Sigmoid()
        )

        for m in self.NN.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)
                
        self.beta = 2.3
        self.gamma = 1
        # self.beta = nn.Parameter(torch.tensor(1.).to(device), requires_grad=True)  ## initial value matters, if we choose 1.5 then it fails
        # self.gamma = nn.Parameter(torch.tensor(1.).to(device), requires_grad=True)
        
    def forward(self, t, y, integro):
        S, I, R = torch.split(y,1,dim=1)
        # print('asfafasfasfsafasfd', I.shape, integro.shape)

        dSdt = -self.beta * S * I + integro
        dIdt = self.beta * S * I - self.gamma * I
        # dIdt = self.NN(torch.cat((S,R),1))
        dRdt = self.gamma * I - integro#- self.memory(I)#sum(pre*dist)*dx
        
        # print('asdf', integro)
        return torch.cat((dSdt,dIdt,dRdt),1)
    
    def integration(self, solution, K, dt):
        # print('sdfsfdsfsf', solution.shape, K.shape)
        
        S, I, R = torch.split(solution, 1, dim=2)
        # https://discuss.pytorch.org/t/one-of-the-variables-required-has-been-modified-by-inplace-operation/104328
        I = I.clone().transpose(1,0)  ######clone ???????
        # print('sfaf: ', I.shape, K.shape)

        integro = I*K
        # print('safdasfasfd', integro.shape)
        integro = torch.sum(integro, dim=1)*dt
        return integro
    
if __name__ == '__main__':
    Erlang = False
    
    if Erlang==True:
        data = np.load('../data/train_sir_l.npy')
        dist = np.load('../data/dist_l.npy')
        t = torch.linspace(0., 15, 1000).to(device)
    else:
        data = np.load('../data/train_sir_l_norm.npy')
        dist = np.load('../data/dist_l_norm.npy')
        t = torch.linspace(0., 25, 100).to(device)

    k = 1
    t = t[::k]
    data = data[:, ::k, :]
    
    method = 'dopri5' ## 'euler'
    # data = np.load('../data/train_sir.npy')
    # k = 5
    # t = torch.linspace(0., 80./k, 200//k).to(device)
    
    func_m = Memory().to(device)
    func = ODEFunc().to(device)
    y = torch.tensor(data, dtype=torch.float32).to(device)
    
    y0 = y[[0],0,:].to(device)
    pred_y = odeint(func, func_m, y0, t, method=method).to(device)
    # pred_y = odeint(func, func_m, y0, t, method='euler').to(device)
    plt.plot(pred_y[:,0,:].cpu().detach(), label=['S', 'I', 'R'])
    plt.plot(data[0])
    plt.legend()
    
    
    
    # optimizer = optim.Adam(func.parameters(), lr=1e-3)
    optimizer = optim.Adam([
                    {'params': func.parameters()},
                    {'params': func_m.parameters(), 'lr': 1e-3}
                ], lr=1e-4)
    
    batch_size = 1
    batch = data
    batch_y = torch.tensor(batch, dtype=torch.float32).to(device)
    batch_y0 = batch_y[:,0,:].to(device)
    
    batch_t = t
    for itr in range(1, 30000):
        # idx = np.random.choice(np.arange(data.shape[0]),batch_size)
        idx = np.array([0])
        batch_y = torch.tensor(data[idx, ...], dtype=torch.float32).to(device)
        batch_y0 = batch_y[:,0,:].to(device)

        optimizer.zero_grad()
        pred_y = odeint(func, func_m, batch_y0, batch_t, method=method).to(device)
        # pred_y = odeint(func, func_m, batch_y0, batch_t, method='euler').to(device)
        pred_y = pred_y.transpose(1,0)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()
        
        if itr%1==0:
            print(f'itr: {itr}, loss: {loss.item():.6f}')
            try:
                print(f'beta: {func.beta.item():.3f}, gamma: {func.gamma.item():.3f}')
            except:
                continue
            
    ## unknown dist; unknown dIdt as neural network; known gamma beta
    torch.save(func_m.state_dict(), f'../models/func_m_N_{device.type}.pt') ### train 1800 iters, loss=0.001
    torch.save(func.state_dict(), f'../models/func_N_{device.type}.pt')
    
    # ### unknown dist; known dIdt; unknown gamma beta
    # torch.save(func_m.state_dict(), f'../models/func_m_p_{device.type}.pt') ### train 26000 iters, loss=0.001
    # torch.save(func.state_dict(), f'../models/func_p_{device.type}.pt')

    # ## unknown dist; known dIdt; known gamma beta
    # torch.save(func_m.state_dict(), f'../models/func_m_{device.type}.pt') ### train 7000 iters, loss=0.001
    # torch.save(func.state_dict(), f'../models/func_{device.type}.pt') 
    
    
    
    
    func_m.load_state_dict(torch.load(f'../models/func_m_N_{device.type}.pt'))
    func.load_state_dict(torch.load(f'../models/func_N_{device.type}.pt'))
    
    # func_m.load_state_dict(torch.load(f'../models/func_m_p_{device.type}.pt')) 
    # func.load_state_dict(torch.load(f'../models/func_p_{device.type}.pt'))
    
    # func_m.load_state_dict(torch.load(f'../models/func_m_{device.type}.pt'))
    # func.load_state_dict(torch.load(f'../models/func_{device.type}.pt'))


    idx = np.random.choice(np.arange(data.shape[0]),batch_size)
    batch_y = torch.tensor(data[idx, ...], dtype=torch.float32).to(device)
    batch_y0 = batch_y[:,0,:].to(device)
    
    pred_y = odeint(func, func_m, batch_y0, batch_t, method=method).to(device)
    # pred_y = odeint(func, func_m, batch_y0, batch_t, method='euler').to(device)
    pred_y = pred_y.transpose(1,0)
    
    fig, ax = plt.subplots(1,2,figsize=(10,4))
    ax[0].plot(pred_y[0].detach().cpu())
    ax[0].plot(batch_y[0].detach().cpu())

    

    K = func_m(t.reshape(-1,1))
    ax[1].plot(K.detach().cpu().numpy()[::-1], label='dist pred')
    ax[1].plot(dist[::k], label='dist')
    ax[1].legend()
    
    # fig.savefig('./figures/unkonw_dist.png')
    
    