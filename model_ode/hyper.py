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
import hyperopt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def hyper_min_2(func, func_m, batch_t, batch_y, method, range_, max_evals=100):
    
    # define an objective function
    def objective(args):
        sigma, mu, beta, S0 = \
            args['sigma'], args['mu'], args['beta'], args['S0']
        # tau = args['tau']
        
        func_m.sigma = nn.Parameter(torch.tensor(sigma).to(device), requires_grad=True)
        func_m.mu = nn.Parameter(torch.tensor(mu).to(device), requires_grad=True)
        func.beta = nn.Parameter(torch.tensor(beta).to(device), requires_grad=True)
        # func.tau = tau
        
        ### initial value of dynamic
        I0 = batch_y[:,0,1].to(device)
        func.S0 = nn.Parameter(torch.tensor([S0]).to(device), requires_grad=True)
        R0 = nn.Parameter(torch.tensor([func.N-func.S0.item()-I0.item()]).to(device), requires_grad=True)
        batch_y0 = torch.cat([func.S0,I0,R0]).reshape(1,3)

        pred_y = odeint(func, func_m, batch_y0, batch_t, method=method).to(device)
        pred_y = pred_y.transpose(1,0)
        
        ### loss
        loss_fn = nn.MSELoss()
        pred_I = pred_y[:,:,1]
        batch_I = batch_y[:,:,1]
        loss = loss_fn(pred_I, batch_I)
        
        ### weight loss
        lll = pred_I.shape[1]
        weight = torch.exp(torch.linspace(0,3,lll)).to(device)
        loss_weighted = torch.square(pred_I-batch_I)# * weight
        loss = loss_weighted.mean()
        
        return loss.item()
    
    # define a search space
    from hyperopt import hp
    
    space = {}
    space['sigma'] = hp.uniform('sigma', range_[0], range_[1])
    space['mu'] = hp.uniform('mu', range_[2], range_[3])
    # space['beta'] = hp.uniform('beta', .5, 10.)
    space['beta'] = hp.uniform('beta', -.8, .2)
    space['S0'] = hp.uniform('S0', 0., 1.*func.N-batch_y[:,0,1].item())

    # space['tau'] = hp.uniform('tau', .7, 1.3)
    
    # minimize the objective over the space
    from hyperopt import fmin, tpe
    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals)

    print(best)
    # print(hyperopt.space_eval(space, best))

    return best




def hyper_min_3(func, func_m, batch_t, batch_y, method, init, range_, max_evals=100):
    
    # define an objective function
    def objective(args):
        # print(args)
        sigma, mu, S0 = args['sigma'], args['mu'], args['S0']
        # tau = args['tau']
        
        func_m.sigma = nn.Parameter(torch.tensor(sigma).to(device), requires_grad=True)
        func_m.mu = nn.Parameter(torch.tensor(mu).to(device), requires_grad=True)
        # func.tau = tau

        ### initial value
        I0 = batch_y[:,0,1].to(device)
        func.S0 = nn.Parameter(torch.tensor([S0]).to(device), requires_grad=True)
        R0 = nn.Parameter(torch.tensor([func.N-func.S0.item()-I0.item()]).to(device), requires_grad=True)
        batch_y0 = torch.cat([func.S0,I0,R0]).reshape(1,3)

        pred_y = odeint(func, func_m, batch_y0, batch_t, method=method).to(device)
        pred_y = pred_y.transpose(1,0)
        
        ### loss
        loss_fn = nn.MSELoss()
        pred_I = pred_y[:,:,1]
        batch_I = batch_y[:,:,1]
        loss = loss_fn(pred_I, batch_I)
        
        ### weight loss
        lll = pred_I.shape[1]
        weight = torch.exp(torch.linspace(0,3,lll)).to(device)
        loss_weighted = torch.square(pred_I-batch_I)# * weight
        loss = loss_weighted.mean()
        
        return loss.item()
    
    # define a search space
    from hyperopt import hp
    
    space = {}
    space['sigma'] = hp.uniform('sigma', range_[0], range_[1])
    space['mu'] = hp.uniform('mu', range_[2], range_[3])
    space['S0'] = hp.uniform('S0', 0., 1.*func.N-batch_y[:,0,1].item())
    ## .7 means around 180 days for 1 wave, 1.5 means 90 days for 1 wave. 1 wave for south korea is 140 days
    
    # space['tau'] = hp.uniform('tau', .7, 1.3)  

    # minimize the objective over the space
    from hyperopt import fmin, tpe
    init_vals = [{'sigma':init[0], 'mu':init[1], 'S0':init[2]}]#, 'tau':init[3]}]
    
    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, points_to_evaluate=init_vals)


    print(best)
    # print(hyperopt.space_eval(space, best))

    return best





