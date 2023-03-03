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
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
                
        
        self.mu = nn.Parameter(torch.tensor(4.5).to(device), requires_grad=True)  ## initial value matters, if we choose 1.5 then it fails
        self.sigma = nn.Parameter(torch.tensor(1.).to(device), requires_grad=True)
        
    # def forward(self, t):
    #     return self.memory(t)
    
    def forward(self, t):
        return 1/(self.sigma*(2*torch.pi)**.5)*torch.exp(-1/2*(t-self.mu)**2/self.sigma**2)


# Erlange = False
# if Erlange==True:    
#     t = torch.linspace(0., 15, 100).to(device)
#     dist = np.load('../data/dist_l.npy')
# else:
#     t = torch.linspace(0., 25, 100).to(device)
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

        self.NN_S = nn.Sequential(
            nn.Linear(3, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
            # ,nn.Sigmoid()
        )
        self.NN_I = nn.Sequential(
            nn.Linear(3, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
            # ,nn.Sigmoid()
        )
        self.NN_R = nn.Sequential(
            nn.Linear(3, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
            # ,nn.Sigmoid()
        )
        for m in self.NN_S.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)
        for m in self.NN_I.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)
        for m in self.NN_R.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)
                
        # self.beta = 2.3
        # self.gamma = 1
        # self.beta = nn.Parameter(torch.tensor(2.0).to(device), requires_grad=True)  ## initial value matters, if we choose 1.5 then it fails
        # self.gamma = nn.Parameter(torch.tensor(1.).to(device), requires_grad=True)
        
        self.NN_beta = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
            # ,nn.Sigmoid()
        )
        
        self.NN_gamma = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
            # ,nn.Sigmoid()
        )
        self.S0 = nn.Parameter(torch.tensor(.99).to(device), requires_grad=True)
                
        
    def forward(self, t, y, integro):
        t = t.reshape([1,1])
        # print(f'asdfasdf:{t.shape}')
        S, I, R = torch.split(y,1,dim=1)
        # print('asfafasfasfsafasfd', I.shape, integro.shape)
        
        dSdt = -self.NN_beta(t) * S * I + integro
        # dSdt = self.NN_S(torch.cat((S,I,R),1)) + integro
        dIdt = self.NN_beta(t) * S * I - self.NN_gamma(t) * I
        # dIdt = self.NN_I(torch.cat((S,I,R),1))
        dRdt = self.NN_gamma(t) * I - integro
        # dRdt = self.NN_R(torch.cat((S,I,R),1)) + integro
        
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
        t = torch.linspace(0., 15, 100).to(device)
    else:
        # data = np.load('../data/train_sir_l_norm.npy') + np.random.randn(100,3)*0.02
        data = np.load('../data/train_sir_l_norm.npy')
        dist = np.load('../data/dist_l_norm.npy')
        t = torch.linspace(0., 25, 100).to(device)
        
    k = 1
    t = t[::k]
    data = data[:, ::k, :]
    
    
    prediction = True
    if prediction==True:
        t = t[:40]
        data = data[:,:40, :]
    
    method = 'euler'##'dopri5' ##
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
    
    
    
    batch_size = 1
    batch = data
    batch_y = torch.tensor(batch, dtype=torch.float32).to(device)
    batch_y0 = batch_y[:,0,:].to(device)
    
    batch_t = t
    
    from hyper import hyper_min, hyper_min_1, hyper_min_2
    for kk in range(20):
        only_I = True        

        if kk!=20:
            print(f'mu: {func_m.mu.item()}, sigma: {func_m.sigma.item()}')
            import copy
            c_func = copy.deepcopy(func)
            c_func_m = copy.deepcopy(func_m)
            ### hyperopt
            # best = hyper_min(func, func_m, batch_t, batch_y, method, only_I)
            # sigma, mu = best['sigma'], best['mu']
            # func_m.sigma = nn.Parameter(torch.tensor(sigma).to(device), requires_grad=True)
            # func_m.mu = nn.Parameter(torch.tensor(mu).to(device), requires_grad=True)
            
            # best = hyper_min_1(func, func_m, batch_t, batch_y, method)
            # sigma, mu, beta, gamma = best['sigma'], best['mu'], best['beta'], best['gamma']
            # func_m.sigma = nn.Parameter(torch.tensor(sigma).to(device), requires_grad=True)
            # func_m.mu = nn.Parameter(torch.tensor(mu).to(device), requires_grad=True)
            # func.beta = nn.Parameter(torch.tensor(beta).to(device), requires_grad=True)
            # func.gamma = nn.Parameter(torch.tensor(gamma).to(device), requires_grad=True)
            
            best = hyper_min_2(c_func, c_func_m, batch_t, batch_y, method, max_evals=50)
            sigma, mu, beta, gamma, S0 = best['sigma'], best['mu'], best['beta'], best['gamma'], best['S0']
            func_m.sigma = nn.Parameter(torch.tensor(sigma).to(device), requires_grad=True)
            func_m.mu = nn.Parameter(torch.tensor(mu).to(device), requires_grad=True)
            func.beta = nn.Parameter(torch.tensor(beta).to(device), requires_grad=True)
            func.gamma = nn.Parameter(torch.tensor(gamma).to(device), requires_grad=True)
            
            # func.S0 = nn.Parameter(torch.tensor(S0).to(device), requires_grad=True)
            I0 = batch_y[:,0,1].to(device)
            batch_y0 = torch.cat([torch.tensor([S0], dtype=torch.float32).to(device),I0,1-S0-I0]).reshape(1,3)

        
        # func_m.sigma = nn.Parameter(torch.tensor(1.).to(device), requires_grad=True)
        # optimizer = optim.Adam(func.parameters(), lr=1e-3)
        optimizer = optim.Adam([
                        {'params': func.parameters()},
                        {'params': func_m.parameters(), 'lr': 1e-3}
                    ], lr=1e-4)
        
        for itr in range(1, 2000):
            # idx = np.random.choice(np.arange(data.shape[0]),batch_size)
            idx = np.array([0])
            batch_y = torch.tensor(data[idx, ...], dtype=torch.float32).to(device)
            if 'batch_y0' not in locals():
                batch_y0 = batch_y[:,0,:].to(device)
    
            optimizer.zero_grad()
            pred_y = odeint(func, func_m, batch_y0, batch_t, method=method).to(device)
            # pred_y = odeint(func, func_m, batch_y0, batch_t, method='euler').to(device)
            pred_y = pred_y.transpose(1,0)
            
        
            if only_I==False:
                loss = torch.mean(torch.abs(pred_y - batch_y))
            else:
                pred_I = pred_y[:,:,1]
                batch_I = batch_y[:,:,1]
                loss = torch.mean(torch.abs(pred_I - batch_I))
    
            
            loss.backward()
            optimizer.step()
            
            if itr%100==0:
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

    
    if prediction==True:
        t = torch.linspace(0., 25, 100).to(device)
        # data = np.load('../data/train_sir_l_norm.npy')

    idx = np.random.choice(np.arange(data.shape[0]),batch_size)
    batch_y = torch.tensor(data[idx, ...], dtype=torch.float32).to(device)
    batch_y0 = batch_y[:,0,:].to(device)
    
    
    # func.S0 = nn.Parameter(torch.tensor(S0).to(device), requires_grad=True)
    I0 = batch_y[:,0,1].to(device)
    batch_y0 = torch.cat([torch.tensor([func.S0.item()], dtype=torch.float32).to(device),I0,abs(1-func.S0.item()-I0)]).reshape(1,3)


    pred_y = odeint(func, func_m, batch_y0, t, method=method).to(device)
    # pred_y = odeint(func, func_m, batch_y0, t, method='euler').to(device)
    pred_y = pred_y.transpose(1,0)
    
    fig, ax = plt.subplots(1,2,figsize=(10,4))
    ax[0].plot(pred_y[0].detach().cpu())
    ax[0].plot(batch_y[0].detach().cpu())

    

    K = func_m(t.reshape(-1,1))
    ax[1].plot(K.detach().cpu().numpy()[::-1], label='dist pred')
    ax[1].plot(dist[::k], label='dist')
    ax[1].legend()
    
    # fig.savefig('./figures/unkonw_dist.png')
    
    