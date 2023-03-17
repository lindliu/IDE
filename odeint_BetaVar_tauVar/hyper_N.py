#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 11:02:00 2023

@author: dliu
"""

# from _impl import odeint
from _impl_origin import odeint

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import hyperopt

def hyper_min_2(func, func_m, batch_t, inter_t, batch_y, method, range_, max_evals=100, need_inter=True):
    
    # define an objective function
    def objective(args):
        # print(args)
        sigma, mu, beta, gamma, S0, tau = \
            args['sigma'], args['mu'], args['beta'], args['gamma'], args['S0'], args['tau']
        # S0 = S0
        
        func_m.sigma = nn.Parameter(torch.tensor(sigma).to(device), requires_grad=True)
        func_m.mu = nn.Parameter(torch.tensor(mu).to(device), requires_grad=True)
        func.beta = nn.Parameter(torch.tensor(beta).to(device), requires_grad=True)
        func.gamma = nn.Parameter(torch.tensor(gamma).to(device), requires_grad=True)
        func.tau = tau
        
        I0 = batch_y[:,0,1].to(device)
        func.S0 = nn.Parameter(torch.tensor([S0]).to(device), requires_grad=True)
        # R0 = nn.Parameter(torch.tensor([func.N-func.S0.item()*func.N-I0.item()]).to(device), requires_grad=True)
        R0 = nn.Parameter(torch.tensor([func.N-func.S0.item()-I0.item()]).to(device), requires_grad=True)

        # batch_y0 = torch.cat([func.S0*func.N,I0,R0]).reshape(1,3)
        batch_y0 = torch.cat([func.S0,I0,R0]).reshape(1,3)

        # idx = np.array([0])
        # batch_y = torch.tensor(data[idx, ...], dtype=torch.float32).to(device)

        
        pred_y = odeint(func, func_m, batch_y0, batch_t, method=method).to(device)
        pred_y = pred_y.transpose(1,0)
        
        loss_fn = nn.MSELoss()
        
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
        
        
        lll = pred_I.shape[1]
        weight = torch.exp(torch.linspace(0,3,lll)).to(device)
        loss_weighted = weight * torch.square(pred_I-batch_I)
        loss = loss_weighted.mean()
        
        return loss.item()
    
    # define a search space
    from hyperopt import hp
    
    space = {}
    space['sigma'] = hp.uniform('sigma', range_[0], range_[1])
    space['mu'] = hp.uniform('mu', range_[2], range_[3])
    space['beta'] = hp.uniform('beta', .5, 15.)
    space['gamma'] = hp.uniform('gamma', .99, 1.)
    space['S0'] = hp.uniform('S0', 0., 1.*func.N)

    space['tau'] = hp.uniform('tau', .5, 1.3)
    
    # minimize the objective over the space
    from hyperopt import fmin, tpe
    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals)


    print(best)
    # print(hyperopt.space_eval(space, best))

    return best




def hyper_min_3(func, func_m, batch_t, inter_t, batch_y, method, init, range_, max_evals=100, need_inter=True):
    
    # define an objective function
    def objective(args):
        # print(args)
        sigma, mu, S0, tau = args['sigma'], args['mu'], args['S0'], args['tau']
        # S0 = S0*func.N
        
        func_m.sigma = nn.Parameter(torch.tensor(sigma).to(device), requires_grad=True)
        func_m.mu = nn.Parameter(torch.tensor(mu).to(device), requires_grad=True)
        # func.beta = nn.Parameter(torch.tensor(beta).to(device), requires_grad=True)
        # func.gamma = nn.Parameter(torch.tensor(gamma).to(device), requires_grad=True)
        func.tau = tau

        I0 = batch_y[:,0,1].to(device)
        func.S0 = nn.Parameter(torch.tensor([S0]).to(device), requires_grad=True)
        # R0 = nn.Parameter(torch.tensor([func.N-func.S0.item()*func.N-I0.item()]).to(device), requires_grad=True)
        R0 = nn.Parameter(torch.tensor([func.N-func.S0.item()-I0.item()]).to(device), requires_grad=True)

        batch_y0 = torch.cat([func.S0,I0,R0]).reshape(1,3)

        # idx = np.array([0])
        # batch_y = torch.tensor(data[idx, ...], dtype=torch.float32).to(device)

        
        pred_y = odeint(func, func_m, batch_y0, batch_t, method=method).to(device)
        pred_y = pred_y.transpose(1,0)
        
        loss_fn = nn.MSELoss()
        
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
        
        # ll = pred_I.shape[1]//3
        # loss1 = torch.sum((pred_I[:,:ll]-batch_I[:,:ll])**2)
        # loss2 = torch.sum((pred_I[:,-ll:]-batch_I[:,-ll:])**2)
        # loss = .3*loss1 + loss2
        
        
        lll = pred_I.shape[1]
        weight = torch.exp(torch.linspace(0,3,lll)).to(device)
        loss_weighted = weight * torch.square(pred_I-batch_I)
        loss = loss_weighted.mean()
        
        return loss.item()
    
    # define a search space
    from hyperopt import hp
    
    space = {}
    space['sigma'] = hp.uniform('sigma', range_[0], range_[1])
    space['mu'] = hp.uniform('mu', range_[2], range_[3])
    # space['beta'] = hp.uniform('beta', .5, 10.)
    # space['gamma'] = hp.uniform('gamma', 0.1, 3.5)
    space['S0'] = hp.uniform('S0', 0., 1.*func.N)
    ## .7 means around 180 days for 1 wave, 1.5 means 90 days for 1 wave. 1 wave for south korea is 140 days
    space['tau'] = hp.uniform('tau', .5, 1.3)  

    # minimize the objective over the space
    from hyperopt import fmin, tpe
    init_vals = [{'sigma':init[0], 'mu':init[1], 'S0':init[2], 'tau':init[3]}]
    
    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, points_to_evaluate=init_vals)


    print(best)
    # print(hyperopt.space_eval(space, best))

    return best





