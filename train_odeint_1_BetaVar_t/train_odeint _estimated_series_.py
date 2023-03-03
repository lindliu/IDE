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
import pandas as pd
import copy

# from _impl import odeint
from _impl_origin import odeint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

class Memory(nn.Module):    
    def __init__(self):
        super(Memory, self).__init__()

        self.mu = nn.Parameter(torch.tensor(4.5).to(device), requires_grad=True)  ## initial value matters, if we choose 1.5 then it fails
        self.sigma = nn.Parameter(torch.tensor(1.).to(device), requires_grad=True)
        
    def forward(self, t):
        return 1/(self.sigma*(2*torch.pi)**.5)*torch.exp(-1/2*(t-self.mu)**2/self.sigma**2)


class ODEFunc1(nn.Module):

    def __init__(self):
        super(ODEFunc1, self).__init__()
        # # self.beta = 2.3
        # # self.gamma = 1
        self.beta = nn.Parameter(torch.tensor(1.3).to(device), requires_grad=True)  ## initial value matters, if we choose 1.5 then it fails
        self.gamma = nn.Parameter(torch.tensor(1.).to(device), requires_grad=True)
        
        self.S0 = nn.Parameter(torch.tensor(.9).to(device), requires_grad=True)
                
    def forward(self, t, y, integro):
        t = t.reshape([1,1])
        S, I, R = torch.split(y,1,dim=1)
    
        dSdt = -self.beta * S * I + integro    
        dIdt = self.beta * S * I - I
        dRdt = I - integro
        
        return torch.cat((dSdt,dIdt,dRdt),1)

class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

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
    
        dSdt = -self.NN_beta(t) * S * I + integro
        dIdt = self.NN_beta(t) * S * I - I
        dRdt = I - integro
        
        return torch.cat((dSdt,dIdt,dRdt),1)

def save_fig(func, func_m, file_name, iteration, loss, length=300):
    T = torch.linspace(0., t_end, length*mul).to(device)
    T_ = torch.linspace(0., t_end, length).to(device)

    # idx = np.random.choice(np.arange(data.shape[0]),batch_size)
    # batch_y = torch.tensor(data[idx, ...], dtype=torch.float32).to(device)
    # batch_y0 = batch_y[:,0,:].to(device)
    
    # func.S0 = nn.Parameter(torch.tensor(S0).to(device), requires_grad=True)
    I0 = batch_y[:,0,1].to(device)
    batch_y0 = torch.cat([torch.tensor([func.S0.item()], dtype=torch.float32).to(device),I0,abs(1-func.S0.item()-I0)]).reshape(1,3)


    pred_y = odeint(func, func_m, batch_y0, T, method=method).to(device)
    # pred_y = odeint(func, func_m, batch_y0, batch_t, method='euler').to(device)
    pred_y_ = pred_y.transpose(1,0)
    pred_y = pred_y_.detach().cpu().numpy()
    
    fig, ax = plt.subplots(2,3,figsize=(10,8))
    ax = ax.flatten()
    ax[0].plot(T, pred_y[0,:,0], label='S predict')
    ax[0].legend()
    ax[0].set_title('S')
    

    # pred_y = odeint(func, func_m, batch_y0, T, method=method).to(device)
    from _impl_origin.misc import Perturb, RegularGridInterpolator
    points_to_interp = [T_]
    I_inter = RegularGridInterpolator([T], pred_y_[:,:,1].flatten())
    pred_I = I_inter(points_to_interp)
    
    # ax[1].plot(pred_y[0,:,1], label=f'I predict {loss:.2e}')
    ax[1].plot(pred_I.detach().cpu(), label=f'I predict {loss:.2e}')
    ax[1].plot(data_[0,:,1], label='I data')
    ax[1].plot(batch_y[0,:,1].detach().cpu(), label='I train')
    ax[1].legend()
    ax[1].set_title('I')

    ax[2].plot(pred_y[0,:,2], label='R predict')
    ax[2].legend()
    ax[2].set_title('R')


    K = func_m(T.reshape(-1,1)).detach().cpu().numpy()[::-1]
    from scipy.stats import norm
    dist = norm.pdf(np.linspace(0,length,10000), loc=70, scale=1)
    ax[3].plot(np.linspace(0,length,10000), dist[::-1], label='dist')
    ax[3].plot(K, label='dist pred')
    ax[3].legend()
    ax[3].set_title('K')

    
    beta = func.NN_beta(T.reshape([-1,1])).detach().cpu().numpy()
    ax[4].plot(beta)
    ax[4].set_title('beta')
        
    os.makedirs(f'./figures/{file_name}',exist_ok=True)
    np.savez(f'./figures/{file_name}/{iteration}.npz', train=batch_y.cpu().numpy(), pred=pred_y, K=K, beta=beta)
    fig.savefig(f'./figures/{file_name}/{iteration}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

def get_train_data(data, start, length, recovery_time, estimate=True, scale=1, data_type='cases_mean'):
    """data_type: 'daily_cases or cases_mean"""
    if estimate==True:
        data_ = data['proportion'][start:start+length].to_numpy().reshape([1,-1,1])
    else:
        cases_convolved = np.convolve(recovery_time*[1], data[data_type], mode='same') / data['population'].iloc[0]
        data_ = cases_convolved[start:start+length].reshape([1,-1,1]) * scale
        
    return data_

def train_beta(func, T, target):
    optimizer = optim.Adam(func.parameters(), lr=1e-3)
    for itr in range(3000):
        optimizer.zero_grad()
        pred = func.NN_beta(T.reshape([-1,1]))
        
        loss = torch.mean(torch.abs(pred - target))
        
        loss.backward()
        optimizer.step()
        
        if itr%500==0:
            print(loss)
    return func

def func_initialization(func, func_m, batch_t, inter_t, batch_y, method, max_evals, need_inter):
    print(f'mu: {func_m.mu.item()}, sigma: {func_m.sigma.item()}')
    c_func = copy.deepcopy(func)
    c_func_m = copy.deepcopy(func_m)
    
    init = [func_m.sigma.item(), func_m.mu.item(), func.S0.item()]
    best = hyper_min_3(c_func, c_func_m, batch_t, inter_t, batch_y, method, init, range_=range_, max_evals=max_evals, need_inter=need_inter)
    sigma, mu, S0 = best['sigma'], best['mu'], best['S0']

    func_m.sigma = nn.Parameter(torch.tensor(sigma).to(device), requires_grad=True)
    func_m.mu = nn.Parameter(torch.tensor(mu).to(device), requires_grad=True)
    
    func.S0 = nn.Parameter(torch.tensor(S0).to(device), requires_grad=True)
    # I0 = batch_y[:,0,1].to(device)
    # batch_y0 = torch.cat([torch.tensor([S0], dtype=torch.float32).to(device),I0,1-S0-I0]).reshape(1,3)
    return func, func_m
    

if __name__ == '__main__':

    countries = ['estimated_United Kingdom', 'estimated_Mexico', 'estimated_Belgium', 
                 'estimated_South Africa', 'estimated_Republic of Korea',\
                 'simulation']
    
    country = countries[-1]
    # country = countries[-1]
    
    ### set false if using real cases to train
    estimate = True
    need_inter = False

    ### load data
    if country!='simulation':
        data = pd.read_csv(f'../data/covid_{country}.csv', sep='\t')
        data['date'] = pd.to_datetime(data['date'])
    
    elif country=='simulation': 
        data = pd.DataFrame(np.load('../data/simulation_2_3.npy'), columns=['S','I','R'])        
        # data["date"] = pd.date_range(start='1/1/2021', periods=500)    
    
    
    dis = 2
    # for num in range(184,250,dis):
    for num in range(220,250,dis):
        writer = SummaryWriter()

        ##### data preparation ######
        length = 400
        recovery_time = 10

        if country == 'estimated_South Africa':
            start = 630
            data_ = get_train_data(data, start, length, recovery_time, estimate)
        elif country == 'estimated_Belgium':
            start = 750
            data_ = get_train_data(data, start, length, recovery_time, estimate)
        elif country == 'estimated_Mexico':
            start = 655
            data_ = get_train_data(data, start, length, recovery_time, estimate, scale=10)
        elif country == 'estimated_United Kingdom':
            start = 750
            data_ = get_train_data(data, start, length, recovery_time, estimate)
        elif country == 'estimated_Republic of Korea':
            start = 710
            data_ = get_train_data(data, start, length, recovery_time, estimate)
            
        elif country=='simulation':
            start = 0
            data_ = data['I'][start:start+length].to_numpy().reshape([1,-1,1])
        
        # plt.plot(data_[0])
        data_ = np.repeat(data_,3,axis=2)
        
        # t_end, mul = 400, 3
        t_end, mul = 25, 1
        T = torch.linspace(0., t_end, length*mul).to(device)
        T_ = torch.linspace(0., t_end, length).to(device)
        
        range_ = [t_end/10, t_end]
        
        # num = 100
        end = start+num
        train_t = copy.deepcopy(T[:num*mul])
        train_data = copy.deepcopy(data_[:,:num, :])
        inter_t = copy.deepcopy(T_[:num])
        
        batch_size = 1
        batch_y = torch.tensor(train_data, dtype=torch.float32).to(device)
        batch_y0 = batch_y[:,0,:].to(device)
        batch_t = train_t
        
        if country == 'simulation':
            file_name = f'{country}_{start}_{end}'
        elif estimate:
            file_name = f'{country}_{start}_{end}'
        else:
            c = country.split('_')[1]
            file_name = f'real_{c}_{start}_{end}'
            

        func = ODEFunc().to(device)        
        func_m = Memory().to(device)
        method = 'euler'##'dopri5' ##
        
        # T = torch.linspace(0., 250, length).to(device)
        # method = 'euler'#'dopri5' ##
        # func = ODEFunc1().to(device)
        # func.beta = nn.Parameter(torch.tensor(2.48).to(device), requires_grad=True)
        # func_m = Memory().to(device)
        # func_m.mu = nn.Parameter(torch.tensor(15.).to(device), requires_grad=True)
        # func_m.sigma = nn.Parameter(torch.tensor(1.).to(device), requires_grad=True)
        # y = torch.tensor(train_data, dtype=torch.float32).to(device)
        # # y0 = y[[0],0,:].to(device)
        # y0 = torch.tensor([[.99,.01,0]], dtype=torch.float).to(device)
        # pred_y = odeint(func, func_m, y0, T, method=method).to(device)
        # plt.plot(pred_y[:,0,:].cpu().detach(), label=['S', 'I', 'R'])
        # # plt.plot(train_data[0])
        # K = func_m(T.reshape(-1,1)).detach().cpu().numpy()[::-1]
        # plt.plot(K)
        # plt.legend()
        
        from hyper import hyper_min_2, hyper_min_3

        ##### find a proper initial value of beta #####
        c_func = ODEFunc1().to(device)
        best = hyper_min_2(c_func, func_m, batch_t, inter_t, batch_y, method=method, range_=range_, max_evals=100, need_inter=need_inter)
        beta_init = best['beta']
        ###############################################

        target = torch.ones(length,1).to(device) * beta_init
        # func = train_beta(func, T, target)
        func = train_beta(func, T_, target)

        for kk in range(10):
            flag = False

            ### initialize mu, sigma and S0 
            func, func_m = func_initialization(func, func_m, batch_t, inter_t, batch_y, method, max_evals=100, need_inter=need_inter)
            
            optimizer = optim.Adam([
                            {'params': func.parameters()},
                            {'params': func_m.parameters(), 'lr': 1e-3}
                        ], lr=1e-4)
            
            loss_fn = nn.MSELoss()##nn.L1Loss()
    
            epoch_sub = 3000
            for itr in range(epoch_sub):
                # idx = np.array([0])
                # batch_y = torch.tensor(train_data[idx, ...], dtype=torch.float32).to(device)
                
                S0 = func.S0.item()
                I0 = batch_y[:,0,1].to(device)
                batch_y0 = torch.cat([torch.tensor([S0], dtype=torch.float32).to(device),I0,1-S0-I0]).reshape(1,3)
                # batch_y0 = batch_y[:,0,:].to(device)
                
                optimizer.zero_grad()
                pred_y = odeint(func, func_m, batch_y0, batch_t, method=method).to(device)
                pred_y = pred_y.transpose(1,0)
                
                
                if need_inter==True:
                    from _impl_origin.misc import Perturb, RegularGridInterpolator
                    points_to_interp = [inter_t]
                    I_inter = RegularGridInterpolator([batch_t], pred_y[:,:,1].flatten())
                    pred_I = I_inter(points_to_interp)
                    
                    batch_I = batch_y[:,:,1]
                    loss = loss_fn(pred_I.flatten(), batch_I.flatten())
                                
                else:
                    pred_I = pred_y[:,:,1]
                    batch_I = batch_y[:,:,1]
                    loss = loss_fn(pred_I, batch_I)
                
                loss.backward()
                optimizer.step()
                
                writer.add_scalar('Loss/train', loss, epoch_sub*kk+itr)

                if itr%100==0:
                    print(f'itr: {epoch_sub*kk+itr}, loss: {loss.item():.2e}')
                    save_fig(func, func_m, file_name, iteration=epoch_sub*kk+itr, loss=loss, length=length)
                    
                    # if loss<5e-06:
                    if loss<4e-05:
                        flag = True
                        break
                    try:
                        print(f'mu: {func_m.mu.item():.2f}, sigma: {func_m.sigma.item():.2f}')
                    except:
                        continue
                
            
            if flag:
                break
            
            
            diff = torch.abs(pred_I-batch_I)
            cop_idx = diff.shape[1]//10  ## first 90% data
            if diff[:,-cop_idx:].sum()/diff.sum()>.3:
                print("retrain beta!!!")
                pred = func.NN_beta(T.reshape([-1,1]))
                target = torch.ones(length,1).to(device) * beta_init
                target[:cop_idx] = pred[:cop_idx].detach()
                func = train_beta(func, T, target)
            

        
        torch.save(func_m.state_dict(), f'./models/func_m_{file_name}_{epoch_sub*kk+itr}_{device.type}.pt') 
        torch.save(func.state_dict(), f'./models/func_{file_name}_{epoch_sub*kk+itr}_{device.type}.pt')
        
        # func_m.load_state_dict(torch.load(f'./models/func_m_{country}_{start}_{end}_{epoch_sub*kk+itr}_{device.type}.pt'))
        # func.load_state_dict(torch.load(f'./models/func_{country}_{start}_{end}_{epoch_sub*kk+itr}_{device.type}.pt'))
    
        writer.close()
        
    writer.flush()

    
    # func_m.load_state_dict(torch.load(f'./models/func_m_simulation_0_126_13501_cuda.pt'))
    # func.load_state_dict(torch.load(f'./models/func_simulation_0_126_13501_cuda.pt'))
    
    # tensorboard --logdir=runs
    