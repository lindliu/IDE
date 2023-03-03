#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:06:35 2022

@author: dliu
"""


import matplotlib.pyplot as plt
# from torchdiffeq import odeint
import numpy as np
import copy
import os
import logging
import pandas as pd

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
            # nn.Sigmoid()
        )
        self.NN_I = nn.Sequential(
            nn.Linear(3, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
            # nn.Sigmoid()
        )
        self.NN_R = nn.Sequential(
            nn.Linear(3, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
            # nn.Sigmoid()
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
                
        # # self.beta = 2.3
        # # self.gamma = 1
        self.beta = nn.Parameter(torch.tensor(2.3).to(device), requires_grad=True)  ## initial value matters, if we choose 1.5 then it fails
        self.gamma = nn.Parameter(torch.tensor(1.).to(device), requires_grad=True)
        
        self.NN_beta = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
            # ,nn.Sigmoid()
            ,nn.Softplus()
        )
        
        for m in self.NN_beta.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1.5)
                nn.init.constant_(m.bias, val=0)
                
        self.S0 = nn.Parameter(torch.tensor(.9).to(device), requires_grad=True)
                
        
    def forward(self, t, y, integro):
        t = t.reshape([1,1])
        S, I, R = torch.split(y,1,dim=1)
        # print('asfafasfasfsafasfd', I.shape, integro.shape)
    
        # dSdt = -self.beta * S * I + integro        
        dSdt = -self.NN_beta(t) * S * I + integro
        # dSdt = self.NN_S(torch.cat((S,I,R),1)) + integro
        # dIdt = self.beta * S * I - self.gamma * I
        dIdt = self.NN_beta(t) * S * I - I
        # dIdt = self.NN_I(torch.cat((S,I,R),1))
        # dRdt = self.gamma * I - integro
        dRdt = I - integro
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
    

def save_fig(func, func_m, country, num, iteration, loss, length=300):
    T = torch.linspace(0., 25, length).to(device)
        
    # idx = np.random.choice(np.arange(data.shape[0]),batch_size)
    # batch_y = torch.tensor(data[idx, ...], dtype=torch.float32).to(device)
    # batch_y0 = batch_y[:,0,:].to(device)
    
    # func.S0 = nn.Parameter(torch.tensor(S0).to(device), requires_grad=True)
    I0 = batch_y[:,0,1].to(device)
    batch_y0 = torch.cat([torch.tensor([func.S0.item()], dtype=torch.float32).to(device),I0,abs(1-func.S0.item()-I0)]).reshape(1,3)


    pred_y = odeint(func, func_m, batch_y0, T, method=method).to(device)
    # pred_y = odeint(func, func_m, batch_y0, batch_t, method='euler').to(device)
    pred_y = pred_y.transpose(1,0)
    
    fig, ax = plt.subplots(2,3,figsize=(10,8))
    ax = ax.flatten()
    # ax[0].plot(pred_y[0].detach().cpu())
    ax[0].plot(pred_y[0,:,0].detach().cpu(), label='S predict')
    # ax[0].plot(data_[0][:,0])
    # ax[0].plot(batch_y[0,:,0].detach().cpu())
    ax[0].legend()
    ax[0].set_title('S')
    
    ax[1].plot(pred_y[0,:,1].detach().cpu(), label=f'I predict {loss:.2e}')
    ax[1].plot(data_[0,:,1], label='I data')
    ax[1].plot(batch_y[0,:,1].detach().cpu(), label='I train')
    ax[1].legend()
    ax[1].set_title('I')

    ax[2].plot(pred_y[0,:,2].detach().cpu(), label='R predict')
    # ax[2].plot(data_[0,:,2])
    # ax[2].plot(batch_y[0,:,2].detach().cpu())
    ax[2].legend()
    ax[2].set_title('R')


    K = func_m(T.reshape(-1,1))
    from scipy.stats import norm
    dist = norm.pdf(np.linspace(0,500,30000), loc=70, scale=1)
    ax[3].plot(K.detach().cpu().numpy()[::-1], label='dist pred')
    ax[3].plot(np.linspace(0,300,300), dist[::-1][::100], label='dist')
    ax[3].legend()
    ax[3].set_title('K')

    
    beta = func.NN_beta(T.reshape([-1,1])).detach().cpu().numpy()
    ax[4].plot(beta)
    ax[4].set_title('beta')
    
    os.makedirs(f'./figures/{country}_{num}',exist_ok=True)
    fig.savefig(f'./figures/{country}_{num}/{iteration}.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    

func = ODEFunc().to(device)
T = torch.linspace(0., 25, 100).to(device)
target = torch.ones(100,1).to(device) * 2

optimizer = optim.Adam(func.parameters(), lr=1e-3)
for itr in range(3000):
    optimizer.zero_grad()
    pred = func.NN_beta(T.reshape([-1,1]))
    
    loss = torch.mean(torch.abs(pred - target))
    
    loss.backward()
    optimizer.step()
    
    if itr%500==0:
        print(loss)


if __name__ == '__main__':

    # func = ODEFunc().to(device)
    func_m = Memory().to(device)
    
    length = 300
    
    countries = ['estimated_United Kingdom', 'estimated_Mexico', 'estimated_Belgium', 
               'estimated_South Africa', 'estimated_Republic of Korea']
    country = countries[4]
    
    data = pd.read_csv(f'../data/covid_{country}.csv', sep='\t')
    data['date'] = pd.to_datetime(data['date'])
    
    if country == 'estimated_South Africa':
        data_ = data['proportion'][630:630+length].to_numpy().reshape([1,-1,1])
    elif country == 'estimated_Belgium':
        data_ = data['proportion'][750:750+length].to_numpy().reshape([1,-1,1])
    elif country == 'estimated_Mexico':
        length = 400
        data_ = data['proportion'][655:655+length].to_numpy().reshape([1,-1,1])
    elif country == 'estimated_United Kingdom':
        data_ = data['proportion'][750:750+length].to_numpy().reshape([1,-1,1])
    elif country == 'estimated_Republic of Korea':
        data_ = data['proportion'][710:710+length].to_numpy().reshape([1,-1,1])

        
    plt.plot(data_[0])
    data_ = np.repeat(data_,3,axis=2)#[:,40:,:]
    # dist = np.load('../data/dist_l_norm.npy')
    T = torch.linspace(0., 25, length).to(device)

    k = 1
    T = T[::k]
    data_ = data_[:, ::k, :]
    
    
    num = 200
    train_t = copy.deepcopy(T[:num])
    train_data = copy.deepcopy(data_[:,:num, :])
    
    
    y = torch.tensor(train_data, dtype=torch.float32).to(device)
    y0 = y[[0],0,:].to(device)
    
    method = 'euler'##'dopri5' ##
    pred_y = odeint(func, func_m, y0, T, method=method).to(device)
    # pred_y = odeint(func, func_m, y0, t, method='euler').to(device)
    plt.plot(pred_y[:,0,:].cpu().detach(), label=['S', 'I', 'R'])
    plt.plot(train_data[0])
    plt.legend()
    
    
    
    ### set logger
    file = os.path.join(f'./models/{country}_{num}_loss.txt')
    # if os.path.exists(file):
    #     os.remove(file)
    logging.basicConfig(filename=file,
                        filemode='w', #'a', #
                        format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
                        # level=logging.DEBUG)
    
    logging.info("iter and loss")
    logger_loss = logging.getLogger('loss')
    # logger_iter = logging.getLogger('iter')
    
        
    batch_size = 1
    batch = train_data
    batch_y = torch.tensor(batch, dtype=torch.float32).to(device)
    batch_y0 = batch_y[:,0,:].to(device)
    
    batch_t = train_t
    
    from hyper import hyper_min, hyper_min_1, hyper_min_2, hyper_min_3
    for kk in range(20):
        only_I = True        

        if kk==0:
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
            
            best = hyper_min_2(c_func, c_func_m, batch_t, batch_y, method, max_evals=100)
            sigma, mu, beta, gamma, S0 = best['sigma'], best['mu'], best['beta'], best['gamma'], best['S0']
            func_m.sigma = nn.Parameter(torch.tensor(sigma).to(device), requires_grad=True)
            func_m.mu = nn.Parameter(torch.tensor(mu).to(device), requires_grad=True)
            func.beta = nn.Parameter(torch.tensor(beta).to(device), requires_grad=True)
            func.gamma = nn.Parameter(torch.tensor(gamma).to(device), requires_grad=True)
            
            func.S0 = nn.Parameter(torch.tensor(S0).to(device), requires_grad=True)
            I0 = batch_y[:,0,1].to(device)
            batch_y0 = torch.cat([torch.tensor([S0], dtype=torch.float32).to(device),I0,1-S0-I0]).reshape(1,3)
            
        else:
            print(f'mu: {func_m.mu.item()}, sigma: {func_m.sigma.item()}')
            import copy
            c_func = copy.deepcopy(func)
            c_func_m = copy.deepcopy(func_m)
            
            init = [func_m.sigma.item(), func_m.mu.item(), func.S0.item()]
            best = hyper_min_3(c_func, c_func_m, batch_t, batch_y, method, init, max_evals=100)
            sigma, mu, S0 = best['sigma'], best['mu'], best['S0']

            func_m.sigma = nn.Parameter(torch.tensor(sigma).to(device), requires_grad=True)
            func_m.mu = nn.Parameter(torch.tensor(mu).to(device), requires_grad=True)
            
            func.S0 = nn.Parameter(torch.tensor(S0).to(device), requires_grad=True)
            I0 = batch_y[:,0,1].to(device)
            batch_y0 = torch.cat([torch.tensor([S0], dtype=torch.float32).to(device),I0,1-S0-I0]).reshape(1,3)

        # func_m.sigma = nn.Parameter(torch.tensor(1.).to(device), requires_grad=True)
        # optimizer = optim.Adam(func.parameters(), lr=1e-3)
        optimizer = optim.Adam([
                        {'params': func.parameters()},
                        {'params': func_m.parameters(), 'lr': 1e-3}
                    ], lr=1e-4)
        
        loss_fn = nn.MSELoss()##nn.L1Loss()

        epoch_sub = 3000
        for itr in range(epoch_sub):
            # idx = np.random.choice(np.arange(data.shape[0]),batch_size)
            idx = np.array([0])
            batch_y = torch.tensor(train_data[idx, ...], dtype=torch.float32).to(device)
            
            if True:
                S0 = func.S0.item()
                I0 = batch_y[:,0,1].to(device)
                batch_y0 = torch.cat([torch.tensor([S0], dtype=torch.float32).to(device),I0,1-S0-I0]).reshape(1,3)

            if 'batch_y0' not in locals():
                batch_y0 = batch_y[:,0,:].to(device)
                
            optimizer.zero_grad()
            pred_y = odeint(func, func_m, batch_y0, batch_t, method=method).to(device)
            # pred_y = odeint(func, func_m, batch_y0, batch_t, method='euler').to(device)
            pred_y = pred_y.transpose(1,0)
            
            
            max_loss = False
            if only_I==False:
                loss = loss_fn(pred_y, batch_y)
            elif max_loss==False:
                pred_I = pred_y[:,:,1]
                batch_I = batch_y[:,:,1]
                loss = loss_fn(pred_I, batch_I)
            elif max_loss==True:
                pred_I = pred_y[:,:,1]
                batch_I = batch_y[:,:,1]
                
                ###https://arxiv.org/pdf/1602.01690.pdf
                idx = torch.argmax(torch.abs(pred_I-batch_I))
                loss = loss_fn(pred_I[0,idx], batch_I[0,idx])
            
            
            loss.backward()
            optimizer.step()
            
            logger_loss.info(f'{epoch_sub*kk+itr} {loss}')

            if itr%500==0:
                print(f'itr: {epoch_sub*kk+itr}, loss: {loss.item():.2e}')
                save_fig(func, func_m, country, num, iteration=epoch_sub*kk+itr, loss=loss, length=length)
                try:
                    print(f'beta: {func.beta.item():.3f}, gamma: {func.gamma.item():.3f}')
                except:
                    continue
            
        if loss<0.0005**2:
            break
    
    
    torch.save(func_m.state_dict(), f'./models/func_m_{country}_{num}_{epoch_sub*kk+itr}_{device.type}.pt') 
    torch.save(func.state_dict(), f'./models/func_{country}_{num}_{epoch_sub*kk+itr}_{device.type}.pt')
    
    func_m.load_state_dict(torch.load(f'./models/func_m_{country}_{num}_{epoch_sub*kk+itr}_{device.type}.pt'))
    func.load_state_dict(torch.load(f'./models/func_{country}_{num}_{epoch_sub*kk+itr}_{device.type}.pt'))
    
    
    
    # func_m.load_state_dict(torch.load(f'./models/func_m_estimated_Belgium_150_10538_cuda.pt'))
    # func.load_state_dict(torch.load(f'./models/func_estimated_Belgium_150_10538_cuda.pt'))
    
    
    