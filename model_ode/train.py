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

import matplotlib
font = {#'family' : 'normal',
        # 'weight' : 'normal', #'bold'
        'size'   : 10}
matplotlib.rc('font', **font)

from hyper import hyper_min_2, hyper_min_3
method = 'midpoint' ## 'euler'##

t_embedding = True#False#
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

class Memory(nn.Module):    
    def __init__(self):
        super(Memory, self).__init__()

        self.mu = nn.Parameter(torch.tensor(4.5).to(device), requires_grad=True)  ## initial value matters, if we choose 1.5 then it fails
        self.sigma = nn.Parameter(torch.tensor(1.).to(device), requires_grad=True)
        
    def forward(self, t):
        return 1/(torch.abs(self.sigma)*(2*torch.pi)**.5)*torch.exp(-1/2*(t-self.mu)**2/torch.abs(self.sigma)**2)


class ODEFunc1(nn.Module):

    def __init__(self, tau=1., lamb=1., N=1):
        super(ODEFunc1, self).__init__()
        # # self.beta = 2.3
        # # self.gamma = 1
        self.beta = nn.Parameter(torch.tensor(1.3).to(device), requires_grad=True)  ## initial value matters, if we choose 1.5 then it fails
        self.gamma = nn.Parameter(torch.tensor(1.).to(device), requires_grad=True)
        
        self.S0 = nn.Parameter(torch.tensor(.9).to(device), requires_grad=True)
        
        self.tau = tau
        self.lamb = lamb
        self.N = N
        
    def forward(self, t, y, integro):
        t = t.reshape([1,1])
        S, I, R = torch.split(y,1,dim=1)
        
        beta = (self.beta + 1) * boundary
        
        dSdt = - self.lamb * beta * S * I / self.N + integro    
        dIdt = self.lamb * beta * S * I / self.N - I
        dRdt = I - integro
        
        return torch.cat((dSdt,dIdt,dRdt),1) * self.tau
    
# https://discuss.pytorch.org/t/set-constraints-on-parameters-or-layers/23620/7
class WeightClipper(object):

    def __init__(self, range_, frequency=5):
        self.range_ = range_
        self.frequency = frequency

    def __call__(self, module):
        # # filter the variables to get the ones you want
        # if hasattr(module, 'weight'):
        w = module.sigma.data
        w = w.clamp(self.range_[0], self.range_[1])
        module.sigma.data = w
        
        w = module.mu.data
        w = w.clamp(self.range_[2], self.range_[3])
        module.mu.data = w
        
class ODEFunc(nn.Module):

    def __init__(self, tau=1., lamb=1., N=1):
        super(ODEFunc, self).__init__()

        # Softsign_cus = SoftsignCustom(c=2)
        
        if t_embedding:
            self.dim_in = 32
        else:
            self.dim_in = 1
            
        self.NN_beta = nn.Sequential(
            nn.Linear(self.dim_in, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1),
            nn.Softsign()
        )
        
        for m in self.NN_beta.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                
        self.S0 = nn.Parameter(torch.tensor(.9).to(device), requires_grad=True)
        
        ### characteristic time step
        self.tau = tau
        self.lamb = lamb
        self.N = N
        
    def forward(self, t, y, integro):
        t = t.reshape([1,1])
        S, I, R = torch.split(y,1,dim=1)
        
        if t_embedding:
            beta = (self.NN_beta(self.time_embedding(t)[:,0,:]) + 1) * boundary
        else:
            beta = (self.NN_beta(t) + 1) * boundary
           
        
        dSdt = - self.lamb * beta * S * I / self.N + integro
        dIdt = self.lamb * beta * S * I / self.N - I
        dRdt = I - integro
        
        return torch.cat((dSdt,dIdt,dRdt),1) * self.tau
    
    
    def time_embedding(self, t):
        d_model = self.dim_in
        pe = torch.zeros(t.shape[0], t.shape[1], d_model).to(device)
        position = t.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
        
        
def save_fig(func, func_m, file_name, iteration, loss, batch_y, N, t_end, data_, error, boundary, length=300):
    T = torch.linspace(0., t_end, length).to(device)
    
    I0 = batch_y[:,0,1].to(device)
    batch_y0 = torch.cat([torch.tensor([func.S0.item()], dtype=torch.float32).to(device),I0,abs(func.N-func.S0.item()-I0)]).reshape(1,3)

    pred_y = odeint(func, func_m, batch_y0, T, method=method).to(device)
    pred_y_ = pred_y.transpose(1,0)
    pred_y = pred_y_.detach().cpu().numpy()
    
    fig, ax = plt.subplots(2,3,figsize=(10,8))
    ax = ax.flatten()
    ax[0].plot(T.cpu().numpy(), pred_y[0,:,0]/N, label='S predict')
    ax[0].legend()
    ax[0].set_title('S')
    
    ax[1].plot(pred_y[0,:,1]/N, label=f'I predict {loss:.2e}')
    ax[1].plot(data_[0,:,1]/N, label='I data')
    ax[1].plot(batch_y[0,:,1].detach().cpu().numpy()/N, label='I train')
    ax[1].plot([], label=f'max error: {error[0]:.2f}')
    ax[1].axvline(error[1],color = 'b')
    ax[1].legend()
    ax[1].set_title('I')

    ax[2].plot(pred_y[0,:,2]/N, label='R predict')
    ax[2].legend()
    ax[2].set_title('R')

    tau = func.tau
    if func_m.sigma.item()<0.03:
        dt_ = abs(func_m.sigma.item())/3
    else:
        dt_ = 0.01
    dt_num = int(T[-1].item()/dt_)+1
    t_ = torch.linspace(0,t_end,dt_num).to(device) * tau

    dt_num_f = int(t_end*1.2/dt_)+1
    t_f = torch.linspace(0,t_end*1.2,dt_num_f).to(device) * func.tau
    normalize_term = func_m(t_f.reshape(-1,1))[1:].sum()*t_f[1].item()
    if device=='cpu':
        normalize_term = normalize_term.detach().numpy()
    else:
        normalize_term = normalize_term.detach().cpu().numpy()
    print(normalize_term)    
    K = func_m(t_.reshape(-1,1)).detach().cpu().numpy()[::-1]/normalize_term
    
    ax[3].plot(t_.cpu()/tau*(length/t_end), K, label='dist pred')
    from scipy.stats import norm
    dist = norm.pdf(np.linspace(0,length,dt_num), loc=70, scale=1)
    ax[3].plot(np.linspace(0,length,dt_num), dist[::-1], label='dist')
    
    sc = (length/t_end/tau)
    ax[3].plot([], label=f'$\mu$: {func_m.mu.item()*sc:.2f}')
    ax[3].plot([], label=f'$\sigma$: {func_m.sigma.item()*sc:.2f}')
    ax[3].plot([], label=f'$\tau$: {tau:.2f}')
    ax[3].plot([], label=f'norm term: {normalize_term:.2f}')
    ax[3].legend()
    ax[3].set_title('K')
    ax[3].set_xticks([0,200,400], [400,200,0])
    
    ### parameters from training
    recovery_time = 14 ### 1/gamma in SIR
    rescale = (length/(t_end*recovery_time))*tau  ### length is days we use
    
    if t_embedding:
        beta = func.NN_beta(func.time_embedding(T.reshape([1,-1])))[0].detach().cpu().numpy()
    else:
        beta = func.NN_beta(T.reshape([-1,1])).detach().cpu().numpy()

    beta = (beta+1)*boundary
    beta = beta/rescale
    ax[4].plot(beta)
    ax[4].plot([], label=f'boundary: {boundary*2}')
    ax[4].legend()
    ax[4].set_title('beta')
    
    os.makedirs(f'./results_save/{file_name}',exist_ok=True)
    np.savez(f'./results_save/{file_name}/{iteration}.npz', train=batch_y.cpu().numpy(), pred=pred_y, K=K, \
             sigma=func_m.sigma.item()*sc, mu=func_m.mu.item()*sc, beta=beta, tau=tau)
    fig.savefig(f'./results_save/{file_name}/{iteration}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

def get_train_data(data, start, length, recovery_time, estimate=True, prop=True, scale=1, data_type='cases_mean'):
    """data_type: 'daily_cases or cases_mean"""
    if estimate==True:
        if prop==True:
            cases_convolved = np.convolve(recovery_time*[1], data['inf_mean'], mode='same')[start:start+length].reshape([1,-1,1])
            data_ = cases_convolved / data['population'].iloc[0]
        else:
            cases_convolved = np.convolve(recovery_time*[1], data['inf_mean'], mode='same')[start:start+length].reshape([1,-1,1])
            data_ = cases_convolved
        
    else:
        if prop==True:
            cases_convolved = np.convolve(recovery_time*[1], data[data_type], mode='same')[start:start+length].reshape([1,-1,1]) * scale
            data_ = cases_convolved / data['population'].iloc[0]
        else:
            cases_convolved = np.convolve(recovery_time*[1], data[data_type], mode='same')[start:start+length].reshape([1,-1,1]) * scale
            data_ = cases_convolved
            
    return data_

def train_beta(func, T, target):
    optimizer = optim.Adam(func.parameters(), lr=1e-3)
    for itr in range(3000):
        optimizer.zero_grad()
        
        if t_embedding:
            pred = func.NN_beta(func.time_embedding(T.reshape([1,-1])))[0]
        else:
            pred = func.NN_beta(T.reshape([-1,1]))
            
        loss = torch.mean(torch.abs(pred - target))
        
        loss.backward()
        optimizer.step()
        
        if itr%500==0:
            print(loss)
    return func

def func_initialization(func, func_m, batch_t, batch_y, method, range_, max_evals):
    print(f'mu: {func_m.mu.item()}, sigma: {func_m.sigma.item()}')
    c_func = copy.deepcopy(func)
    c_func_m = copy.deepcopy(func_m)
    
    sigma_init = np.clip(func_m.sigma.item(), range_[0], range_[1])
    mu_init = np.clip(func_m.mu.item(), range_[2], range_[3])
    init = [sigma_init, mu_init, func.S0.item(), func.tau]
    
    best = hyper_min_3(c_func, c_func_m, batch_t, batch_y, method, init, range_=range_, max_evals=max_evals)
    sigma, mu, S0 = best['sigma'], best['mu'], best['S0']
    
    func_m.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32).to(device), requires_grad=True)
    func_m.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float32).to(device), requires_grad=True)
    
    func.S0 = nn.Parameter(torch.tensor(S0, dtype=torch.float32).to(device), requires_grad=True)
    
    return func, func_m
    

def test():
    method = 'euler' #'midpoint' # 'dopri5' #
    ### func
    func = ODEFunc1(tau=1.).to(device)
    beta = 2.3
    beta = (beta-boundary)/boundary
    func.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32).to(device), requires_grad=True)
    
    ### func_m
    func_m = Memory().to(device)
    func_m.mu = nn.Parameter(torch.tensor(5.).to(device), requires_grad=True)
    func_m.sigma = nn.Parameter(torch.tensor(1.).to(device), requires_grad=True)
    
    length = 400
    ### dynamic
    T = torch.linspace(0., length/14, length).to(device)
    y0 = torch.tensor([[.99,.01,0]], dtype=torch.float).to(device)
    pred_y = odeint(func, func_m, y0, T, method=method).to(device)
    
    ### K
    K = func_m(T.reshape(-1,1)).detach().cpu().numpy()[::-1]

    ### plot
    fig, ax = plt.subplots(1,2,figsize=(10,3))
    ax[0].plot(pred_y[:,0,:].cpu().detach().numpy(), label=['S', 'I', 'R'])
    ax[1].plot(K, label='distrubition K')
    ax[0].legend()
    ax[1].legend()

def main(country, estimate, prop, array, experiments=None, tol=1e-4):
    ### load data
    if country not in ['numerical', 'simulation']:
        data = pd.read_csv(f'../data/covid_{country}.csv', sep='\t')
        data['date'] = pd.to_datetime(data['date'])
    
    elif country=='simulation': 
        data = pd.DataFrame(np.load('../data/simulation_2_3.npy'), columns=['S','I','R'])            
    
    elif country=='numerical':
        data = np.load('../data/numerical.npz')['SIR']
        
    for num in array:
        ##### data preparation ######
        length = 400
        recovery_time = 14  ## gamma=1/14

        if country in ['Mexico', 'South Africa', 'Republic of Korea']:
            start = 640#714#741#690#640#708#
            data_ = get_train_data(data, start, length, recovery_time, estimate, prop)
        
        elif country=='simulation':
            start = 0
            data_ = data['I'][start:start+length].to_numpy().reshape([1,-1,1])
        
        elif country=='numerical':
            start = 0
            data_ = data[0,:,2].reshape([1,-1,1])
        
        
        if prop or country in ['simulation', 'numerical']:
            N = 1
        else:
            N =  int(data['population'].iloc[0])
        
        data_ = np.repeat(data_,3,axis=2)
        
        t_end = length/recovery_time ###25
        T = torch.linspace(0., t_end, length).to(device)
        
        range_ = [1e-3, t_end/10, recovery_time*3/length*t_end, t_end*1.2] ## sigma boundary, mu boundary
        
        end = start+num
        train_t = copy.deepcopy(T[:num])
        train_data = copy.deepcopy(data_[:,:num, :])
        
        batch_size = 1
        batch_y = torch.tensor(train_data, dtype=torch.float32).to(device)
        batch_y0 = batch_y[:,0,:].to(device)
        batch_t = train_t
        
        if country in ['simulation', 'numerical']:
            file_name = f'{country}_{start}_{end}' \
                if experiments is None else f'{country}_{start}_{end}_{experiments}'
        elif estimate:
            file_name = f'estimate_{country}_{start}_{end}' \
                if experiments is None else f'estimate_{country}_{start}_{end}_{experiments}' 
        else:
            file_name = f'real_{country}_{start}_{end}'\
                if experiments is None else f'real_{country}_{start}_{end}_{experiments}' 
       
        writer = SummaryWriter(log_dir=f'./runs/{file_name}')

        
        func = ODEFunc(N=N).to(device)
        func_m = Memory().to(device)        
        clipper = WeightClipper(range_)

        ##### find a proper initial value of beta #####
        c_func = ODEFunc1(N=N).to(device)
        best = hyper_min_2(c_func, func_m, batch_t, batch_y, method=method, \
                           range_=range_, max_evals=300)
        beta_init = best['beta']
        ###############################################
        
        target = torch.ones(length,1).to(device) * beta_init
        func = train_beta(func, T, target)

        count = 0
        for kk in range(15):
            flag = False

            ### initialize mu, sigma and S0 
            func, func_m = func_initialization(func, func_m, batch_t, batch_y, \
                                               method, range_, max_evals=100)
            
            optimizer = optim.Adam([
                            {'params': func.parameters()},
                            {'params': func_m.parameters(), 'lr': 1e-3}
                        ], lr=1e-4)
            
            loss_fn = nn.MSELoss()##nn.L1Loss()
            
            # epoch_sub_ = np.arange(30,591,40)
            # epoch_sub = epoch_sub_[kk]
            epoch_sub = 300
            for itr in range(epoch_sub):
                
                S0 = func.S0.item()
                I0 = batch_y[:,0,1].to(device)
                batch_y0 = torch.cat([torch.tensor([S0], dtype=torch.float32).to(device),I0,func.N-S0-I0]).reshape(1,3)
                
                optimizer.zero_grad()
                pred_y = odeint(func, func_m, batch_y0, batch_t, method=method).to(device)
                pred_y = pred_y.transpose(1,0)
                
                ### loss
                pred_I = pred_y[:,:,1]
                batch_I = batch_y[:,:,1]
                loss = loss_fn(pred_I, batch_I)
                
                ### weight loss
                lll = pred_I.shape[1]
                weight = torch.exp(torch.linspace(0,3,lll)).to(device)
                loss_weighted = torch.square(pred_I-batch_I) #* weight
                loss = loss_weighted.mean()
        
                loss.backward()
                optimizer.step()
                
                func_m.apply(clipper)
                
                writer.add_scalar(f'{file_name}_Loss', loss, count)
                writer.add_scalar(f'{file_name}_mu', func_m.mu.item(), count)
                writer.add_scalar(f'{file_name}_sigma', func_m.sigma.item(), count)
                
                count = count+1
                if (count-1)%100==0:
                    
                    val, ind = torch.max(torch.abs(pred_I-batch_I)/(batch_I+1e-5),dim=1)
                    error, ind = val.detach().item(), ind.item()
                    print(f'relative error: {error:.2f}')
                    
                    
                    print(f'itr: {count-1}, loss: {loss.item():.2e}')
                    save_fig(func, func_m, file_name, iteration=count-1, loss=loss, \
                             batch_y=batch_y, N=N, t_end=t_end, data_=data_, error=[error,ind], \
                             boundary=boundary, length=length)
                        
                    if loss<tol: 
                    # if error<.2:
                        flag = True
                        break
                    try:
                        print(f'mu: {func_m.mu.item():.2f}, sigma: {func_m.sigma.item():.2f}, tau: {func.tau:.2f}')
                    except:
                        continue
                    
            if flag:
                break
        
        # torch.save(func_m.state_dict(), f'./models/func_m_{file_name}_{epoch_sub*kk+itr}_{device.type}.pt')
        # torch.save(func.state_dict(), f'./models/func_{file_name}_{epoch_sub*kk+itr}_{device.type}.pt')
        
        writer.close()
        
    writer.flush()


    # func_m.load_state_dict(torch.load(f'./models/func_m_simulation_0_126_13501_cuda.pt'))
    # func.load_state_dict(torch.load(f'./models/func_simulation_0_126_13501_cuda.pt'))

    # tensorboard --logdir=runs
    
if __name__ == '__main__':

    ### 'real Mexico', 'real South Africa', 'real Korea', 'est Mexico', 'est South Africa', 'est Korea'
    peak_st = np.array([71, 41, 134, 62, 31, 120]) ### first peak
    peak_nd = np.array([247, 187, 283, 240, 177, 273]) ### second peak                
                
                

    
    #############################################
    ### for simulated data and estimated data ###
    #############################################
    
    ### boundary of R0
    boundary = 5
    countries = ['numerical', 'simulation', 'Mexico', 'South Africa', 'Republic of Korea']
    # tol = [1e-4, 1e-4, 1e-5, 1e-5, 5e-6]
    tol = [1e-4, 1e-4, 5e-6, 1e-5, 2e-6]

    ### set estimate=false if using real cases to train
    estimate, prop = True, True 
    for tol_, country in zip(tol[1:], countries[1:]):
        if country=='simulation':
            boundary = 3
            # ### generate results every 6 days and all days 4 weeks before peaks(not include the week that the peak happens)
            array = np.unique([*np.arange(20,350,6),*np.arange(39-7*4-6,39-6),*np.arange(178-7*4-6,178-6),*np.arange(291-7*4-6,291-6)])
            main(country, estimate, prop, array, tol=tol_)
        
        if country=='Mexico':
            boundary = 5
            array = np.unique([*np.arange(20,350,6),*np.arange(peak_st[3]-7*4-6,peak_st[3]-6),*np.arange(peak_nd[3]-7*4-6,peak_nd[3]-6)])
            main(country, estimate, prop, array, tol=tol_)
            
        if country=='South Africa':
            boundary = 5
            array = np.unique([*np.arange(20,350,6),*np.arange(3,               peak_st[4]-6),*np.arange(peak_nd[4]-7*4-6,peak_nd[4]-6)])
            main(country, estimate, prop, array, tol=tol_)
            
        if country=='Republic of Korea':
            boundary = 5
            array = np.unique([*np.arange(20,350,6),*np.arange(peak_st[5]-7*4-6,peak_st[5]-6),*np.arange(peak_nd[5]-7*4-6,peak_nd[5]-6)])
            main(country, estimate, prop, array, tol=tol_)
            
        
    
    # ####################
    # ## for real data ###
    # ####################
    
    # boundary = 5
    # countries = ['Mexico', 'South Africa', 'Republic of Korea']
    # # tol = [8e+8,5e+7,5e+9]
    # tol = [8e+8,5e+7,1e+9]
    
    # estimate, prop = False, False 
    # for tol_, country in zip(tol, countries):
    #     if country=='Mexico':
    #         boundary = 5
    #         array = np.unique([*np.arange(20,350,6),*np.arange(peak_st[0]-7*4-6,peak_st[0]-6),*np.arange(peak_nd[0]-7*4-6,peak_nd[0]-6)])
    #         main(country, estimate, prop, array, tol=tol_)
    
    #     if country=='South Africa':
    #         boundary = 5
    #         array = np.unique([*np.arange(20,350,6),*np.arange(peak_st[1]-7*4-6,peak_st[1]-6),*np.arange(peak_nd[1]-7*4-6,peak_nd[1]-6)])
    #         main(country, estimate, prop, array, tol=tol_)
    
    #     if country=='Republic of Korea':
    #         boundary = 5
    #         array = np.unique([*np.arange(20,350,6),*np.arange(peak_st[2]-7*4-6,peak_st[2]-6),*np.arange(peak_nd[2]-7*4-6,peak_nd[2]-6)])
    #         main(country, estimate, prop, array, tol=tol_)
        
    
    
    
    
            
            
            
    
    ######################### multiple samples over one data period ################################
    
    # ############################################
    # ## for simulated data and estimated data ###
    # ############################################
    
    # ### boundary of R0
    # boundary = 5#3#1.75 #
    # countries = ['numerical', 'simulation', 'Mexico', 'South Africa', 'Republic of Korea']
    # tol = [1e-4, 1e-4, 5e-6, 1e-4, 5e-6]
    
    # ### set estimate=false if using real cases to train
    # estimate, prop = True, True 
    # for tol_, country in zip(tol[1:], countries[1:]):
    #     if country in ['simulation']:
    #         boundary = 1.75
    #         for experiments in range(0,20):
    #             array = np.arange(347,350,3) ##np.arange(20,350,3)
    #             main(country, estimate, prop, array, experiments, tol=tol_)
                
    #         boundary = 3
    #         for experiments in range(20,40):
    #             array = np.arange(347,350,3) ##np.arange(20,350,3)
    #             main(country, estimate, prop, array, experiments, tol=tol_)
                
    #         boundary = 5
    #         for experiments in range(40,60):
    #             array = np.arange(347,350,3) ##np.arange(20,350,3)
    #             main(country, estimate, prop, array, experiments, tol=tol_)
                    
        
    #     if country in ['Mexico']:
    #         boundary = 5
    #         for experiments in range(0,20):
    #             array = np.arange(347,350,3) ##np.arange(20,350,3)
    #             main(country, estimate, prop, array, experiments, tol=tol_)
                
    #     if country in ['South Africa']:
    #         boundary = 5
    #         for experiments in range(0,20):
    #             array = np.arange(347,350,3) ##np.arange(20,350,3)
    #             main(country, estimate, prop, array, experiments, tol=tol_)
            
    #     if country in ['Republic of Korea']:
    #         boundary = 5
    #         for experiments in range(0,20):
    #             array = np.arange(347,350,3) ##np.arange(20,350,3)
    #             main(country, estimate, prop, array, experiments, tol=tol_)
    
    
    
    # ####################
    # ## for real data ###
    # ####################
    
    # boundary = 5
    # countries = ['Mexico', 'South Africa', 'Republic of Korea']
    # tol = [1e+9,1e+8,5e+9]
    
    # estimate, prop = False, False 
    # for tol_, country in zip(tol, countries):
    #     if country in ['Mexico']:
    #         boundary = 5
    #         for experiments in range(19,20):
    #             array = np.arange(347,350,3) ##np.arange(20,350,3)
    #             main(country, estimate, prop, array, experiments, tol=tol_)
    
    #     if country in ['South Africa']:
    #         boundary = 5
    #         for experiments in range(0,20):
    #             array = np.arange(347,350,3) ##np.arange(20,350,3)
    #             main(country, estimate, prop, array, experiments, tol=tol_)
    
    #     if country in ['Republic of Korea']:
    #         boundary = 5
    #         for experiments in range(0,20):
    #             array = np.arange(347,350,3) ##np.arange(20,350,3)
    #             main(country, estimate, prop, array, experiments, tol=tol_)
                
        
