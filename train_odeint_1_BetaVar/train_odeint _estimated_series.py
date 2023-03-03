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


class ODEFunc1(nn.Module):

    def __init__(self):
        super(ODEFunc1, self).__init__()
        # # self.beta = 2.3
        # # self.gamma = 1
        self.beta = nn.Parameter(torch.tensor(2.3).to(device), requires_grad=True)  ## initial value matters, if we choose 1.5 then it fails
        self.gamma = nn.Parameter(torch.tensor(1.).to(device), requires_grad=True)
        
        self.S0 = nn.Parameter(torch.tensor(.9).to(device), requires_grad=True)
                
    def forward(self, t, y, integro):
        t = t.reshape([1,1])
        S, I, R = torch.split(y,1,dim=1)
        # print('asfafasfasfsafasfd', I.shape, integro.shape)
    
        dSdt = -self.beta * S * I + integro        
        # dSdt = -self.NN_beta(t) * S * I + integro
        # dSdt = self.NN_S(torch.cat((S,I,R),1)) + integro
        dIdt = self.beta * S * I - I
        # dIdt = self.NN_beta(t) * S * I - I
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


def save_fig(func, func_m, country, num, iteration, loss, length=300, estimate=True):
    T = torch.linspace(0., 25, length).to(device)
        
    # idx = np.random.choice(np.arange(data.shape[0]),batch_size)
    # batch_y = torch.tensor(data[idx, ...], dtype=torch.float32).to(device)
    # batch_y0 = batch_y[:,0,:].to(device)
    
    # func.S0 = nn.Parameter(torch.tensor(S0).to(device), requires_grad=True)
    I0 = batch_y[:,0,1].to(device)
    batch_y0 = torch.cat([torch.tensor([func.S0.item()], dtype=torch.float32).to(device),I0,abs(1-func.S0.item()-I0)]).reshape(1,3)


    pred_y = odeint(func, func_m, batch_y0, T, method=method).to(device)
    # pred_y = odeint(func, func_m, batch_y0, batch_t, method='euler').to(device)
    pred_y = pred_y.transpose(1,0).detach().cpu().numpy()
    
    fig, ax = plt.subplots(2,3,figsize=(10,8))
    ax = ax.flatten()
    # ax[0].plot(pred_y[0].detach().cpu())
    ax[0].plot(pred_y[0,:,0], label='S predict')
    # ax[0].plot(data_[0][:,0])
    # ax[0].plot(batch_y[0,:,0].detach().cpu())
    ax[0].legend()
    ax[0].set_title('S')
    
    ax[1].plot(pred_y[0,:,1], label=f'I predict {loss:.2e}')
    ax[1].plot(data_[0,:,1], label='I data')
    ax[1].plot(batch_y[0,:,1].detach().cpu(), label='I train')
    ax[1].legend()
    ax[1].set_title('I')

    ax[2].plot(pred_y[0,:,2], label='R predict')
    # ax[2].plot(data_[0,:,2])
    # ax[2].plot(batch_y[0,:,2].detach().cpu())
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
    
    if estimate:
        os.makedirs(f'./figures/{country}_{num}',exist_ok=True)
        np.savez(f'./figures/{country}_{num}/{iteration}.npz', train=batch_y.cpu().numpy(), pred=pred_y, K=K, beta=beta)
        fig.savefig(f'./figures/{country}_{num}/{iteration}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        os.makedirs(f'./figures/real_{country}_{num}',exist_ok=True)
        np.savez(f'./figures/real_{country}_{num}/{iteration}.npz', train=batch_y.cpu().numpy(), pred=pred_y, K=K, beta=beta)
        fig.savefig(f'./figures/real_{country}_{num}/{iteration}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

def get_train_data(data, start, length, recovery_time, estimate=True, scale=1, data_type='cases_mean'):
    """data_type: 'daily_cases or cases_mean"""
    if estimate==True:
        data_ = data['proportion'][start:start+length].to_numpy().reshape([1,-1,1])
    else:
        cases_convolved = np.convolve(recovery_time*[1], data[data_type], mode='same') / data['population'].iloc[0]
        data_ = cases_convolved[start:start+length].reshape([1,-1,1]) * scale
        
    return data_

if __name__ == '__main__':

    # func = ODEFunc().to(device)
    func_m = Memory().to(device)
    
    countries = ['estimated_United Kingdom', 'estimated_Mexico', 'estimated_Belgium', 
                 'estimated_South Africa', 'estimated_Republic of Korea',\
                 'simulation']
    
    country = countries[1]
    # country = countries[-1]
    
    ### set false if using real cases to train
    estimate = False#True 
    
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
        
        length = 400
        recovery_time = 10
        func = ODEFunc().to(device)        
        
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
        
        
        plt.plot(data_[0])
        data_ = np.repeat(data_,3,axis=2)#[:,40:,:]
        # dist = np.load('../data/dist_l_norm.npy')
        T = torch.linspace(0., 25, length).to(device)
    
        k = 1
        T = T[::k]
        data_ = data_[:, ::k, :]
        
        
        # num = 100
        end = start+num
        train_t = copy.deepcopy(T[:num])
        train_data = copy.deepcopy(data_[:,:num, :])
        
        
        y = torch.tensor(train_data, dtype=torch.float32).to(device)
        y0 = y[[0],0,:].to(device)
        
        method = 'euler'##'dopri5' ##
        pred_y = odeint(func, func_m, y0, T, method=method).to(device)
        # plt.plot(pred_y[:,0,:].cpu().detach(), label=['S', 'I', 'R'])
        # plt.plot(train_data[0])
        # plt.legend()
        
        
        ### set logger
        file = os.path.join(f'./models/{country}_{start}_{end}_loss.txt')
        # if os.path.exists(file):
        #     os.remove(file)
        logging.basicConfig(filename=file,
                            filemode='a', #'w', #
                            format='%(asctime)s %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)
                            # level=logging.DEBUG)
        
        logging.info("iter and loss")
        logger_loss = logging.getLogger(f'loss_{num}')
        # logger_iter = logging.getLogger('iter')
        
            
        batch_size = 1
        batch_y = torch.tensor(train_data, dtype=torch.float32).to(device)
        batch_y0 = batch_y[:,0,:].to(device)
        
        batch_t = train_t
        
        
        
        from hyper import hyper_min, hyper_min_1, hyper_min_2, hyper_min_3

        ##### find a proper initial value of beta #####
        c_func = ODEFunc1().to(device)
        # c_func_m = Memory().to(device)
        best = hyper_min_2(c_func, func_m, batch_t, batch_y, method='euler', max_evals=100)
        ###############################################

        # T = torch.linspace(0., 25, length).to(device)
        target = torch.ones(length,1).to(device) * best['beta']
        
        optimizer = optim.Adam(func.parameters(), lr=1e-3)
        for itr in range(3000):
            optimizer.zero_grad()
            pred = func.NN_beta(T.reshape([-1,1]))
            
            loss = torch.mean(torch.abs(pred - target))
            
            loss.backward()
            optimizer.step()
            
            if itr%500==0:
                print(loss)


        for kk in range(10):
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
            
            flag = False
            
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
                    save_fig(func, func_m, country, f'{start}_{end}_', iteration=epoch_sub*kk+itr, loss=loss, length=length, estimate=estimate)
                    
                    if loss<8e-07:
                    # if loss<4e-05:
                        flag = True
                        break
                
                    try:
                        print(f'beta: {func.beta.item():.3f}, gamma: {func.gamma.item():.3f}')
                        print(f'mu: {func_m.mu.item():.2f}, sigma: {func_m.sigma.item():.2f}')
                    except:
                        continue
                
            
            if flag:
                break
        
            
            # if True:
            #     pred = func.NN_beta(T.reshape([-1,1]))
            #     target = torch.ones(length,1).to(device)*5
            #     target[:120] = pred[:120].detach()
                
            #     optimizer = optim.Adam(func.parameters(), lr=1e-3)
            #     for itr in range(5000):
            #         optimizer.zero_grad()
            #         pred = func.NN_beta(T.reshape([-1,1]))
                    
            #         loss = torch.mean(torch.abs(pred - target))
                    
            #         loss.backward()
            #         optimizer.step()
                    
            #         if itr%500==0:
            #             print(loss)
                        
                
        if estimate:
            torch.save(func_m.state_dict(), f'./models/func_m_{country}_{start}_{end}_{epoch_sub*kk+itr}_{device.type}.pt') 
            torch.save(func.state_dict(), f'./models/func_{country}_{start}_{end}_{epoch_sub*kk+itr}_{device.type}.pt')
        else:
            torch.save(func_m.state_dict(), f'./models/func_m_real_{country}_{start}_{end}_{epoch_sub*kk+itr}_{device.type}.pt') 
            torch.save(func.state_dict(), f'./models/func_real_{country}_{start}_{end}_{epoch_sub*kk+itr}_{device.type}.pt')

        # func_m.load_state_dict(torch.load(f'./models/func_m_{country}_{start}_{end}_{epoch_sub*kk+itr}_{device.type}.pt'))
        # func.load_state_dict(torch.load(f'./models/func_{country}_{start}_{end}_{epoch_sub*kk+itr}_{device.type}.pt'))
    
    
    
    # func_m.load_state_dict(torch.load(f'./models/func_m_simulation_0_126_13501_cuda.pt'))
    # func.load_state_dict(torch.load(f'./models/func_simulation_0_126_13501_cuda.pt'))
    
    
    