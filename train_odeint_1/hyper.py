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


# class Memory(nn.Module):    
#     def __init__(self):
#         super(Memory, self).__init__()

#         self.memory = nn.Sequential(
#             nn.Linear(1, 20),
#             nn.Tanh(),
#             nn.Linear(20, 20),
#             nn.Tanh(),
#             nn.Linear(20, 20),
#             nn.Tanh(),
#             nn.Linear(20, 1),
#             # nn.ReLU()
#             nn.Sigmoid()
#         )
        
#         for m in self.memory.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, mean=0, std=0.1)
#                 nn.init.constant_(m.bias, val=0)
                
        
#         self.mu = nn.Parameter(torch.tensor(4.5).to(device), requires_grad=True)  ## initial value matters, if we choose 1.5 then it fails
#         self.sigma = nn.Parameter(torch.tensor(1.).to(device), requires_grad=True)
        
#     # def forward(self, t):
#     #     return self.memory(t)
    
#     def forward(self, t):
#         return 1/(self.sigma*(2*torch.pi)**.5)*torch.exp(-1/2*(t-self.mu)**2/self.sigma**2)


# # Erlange = False
# # if Erlange==True:    
# #     t = torch.linspace(0., 15, 100).to(device)
# #     dist = np.load('../data/dist_l.npy')
# # else:
# #     t = torch.linspace(0., 25, 100).to(device)
# #     dist = np.load('../data/dist_l_norm.npy')

# # dist = torch.tensor(dist, dtype=torch.float32).to(device)

# # batch_t = t.reshape(-1,1)
# # batch_dist = dist.reshape(-1,1)

# # func_m = Memory().to(device)
# # optimizer = optim.RMSprop(func_m.parameters(), lr=1e-3)

# # for itr in range(1, 20000):
# #     optimizer.zero_grad()
    
# #     pred_dist = func_m(batch_t)
# #     pred_dist = torch.flip(pred_dist, dims=(0,))
# #     loss = nn.functional.mse_loss(pred_dist, batch_dist)

# #     loss.backward()
# #     optimizer.step()
    
# #     if itr%1000==0:
# #         print(loss.item())

# # pred_dist = func_m(batch_t)  
# # plt.plot(pred_dist.cpu().detach().numpy()[::-1])
# # plt.plot(batch_dist.cpu())



# class ODEFunc(nn.Module):

#     def __init__(self):
#         super(ODEFunc, self).__init__()

#         self.NN = nn.Sequential(
#             nn.Linear(2, 20),
#             nn.Tanh(),
#             nn.Linear(20, 20),
#             nn.Tanh(),
#             nn.Linear(20, 20),
#             nn.Tanh(),
#             nn.Linear(20, 1)
#             # nn.Sigmoid()
#         )

#         for m in self.NN.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, mean=0, std=0.01)
#                 nn.init.constant_(m.bias, val=0)
                
#         # self.beta = 2.3
#         # self.gamma = 1
#         self.beta = nn.Parameter(torch.tensor(1.8).to(device), requires_grad=True)  ## initial value matters, if we choose 1.5 then it fails
#         self.gamma = nn.Parameter(torch.tensor(1.).to(device), requires_grad=True)
        
#     def forward(self, t, y, integro):
#         S, I, R = torch.split(y,1,dim=1)
#         # print('asfafasfasfsafasfd', I.shape, integro.shape)

#         dSdt = -self.beta * S * I + integro
#         dIdt = self.beta * S * I - self.gamma * I
#         # dIdt = self.NN(torch.cat((S,R),1))
#         dRdt = self.gamma * I - integro#- self.memory(I)#sum(pre*dist)*dx
        
#         # print('asdf', integro)
#         return torch.cat((dSdt,dIdt,dRdt),1)
    
#     def integration(self, solution, K, dt):
#         # print('sdfsfdsfsf', solution.shape, K.shape)
        
#         S, I, R = torch.split(solution, 1, dim=2)
#         # https://discuss.pytorch.org/t/one-of-the-variables-required-has-been-modified-by-inplace-operation/104328
#         I = I.clone().transpose(1,0)  ######clone ???????
#         # print('sfaf: ', I.shape, K.shape)

#         integro = I*K
#         # print('safdasfasfd', integro.shape)
#         integro = torch.sum(integro, dim=1)*dt
#         return integro

# Erlang = False

# if Erlang==True:
#     data = np.load('../data/train_sir_l.npy')
#     dist = np.load('../data/dist_l.npy')
#     t = torch.linspace(0., 15, 100).to(device)
# else:
#     data = np.load('../data/train_sir_l_norm.npy')
#     dist = np.load('../data/dist_l_norm.npy')
#     t = torch.linspace(0., 25, 100).to(device)
        


# k = 1
# t = t[::k]
# data = data[:, ::k, :]


# prediction = True
# if prediction==True:
#     t = t[:40]
#     data = data[:,:40, :]
    
# func_m = Memory().to(device)
# func = ODEFunc().to(device)
    
# batch_size = 1
# batch = data
# batch_y = torch.tensor(batch, dtype=torch.float32).to(device)
# batch_y0 = batch_y[:,0,:].to(device)
# batch_t = t

# method = 'euler'##'dopri5' ##





def hyper_min(func, func_m, batch_t, batch_y, method, only_I=True, max_evals=100):
    
    # define an objective function
    def objective(args):
        # print(args)
        sigma, mu = args['sigma'], args['mu']
        
        func_m.sigma = nn.Parameter(torch.tensor(sigma).to(device), requires_grad=True)
        func_m.mu = nn.Parameter(torch.tensor(mu).to(device), requires_grad=True)
    
        # idx = np.array([0])
        # batch_y = torch.tensor(data[idx, ...], dtype=torch.float32).to(device)
        batch_y0 = batch_y[:,0,:].to(device)
    
        pred_y = odeint(func, func_m, batch_y0, batch_t, method=method).to(device)
        # pred_y = odeint(func, func_m, batch_y0, batch_t, method='euler').to(device)
        pred_y = pred_y.transpose(1,0)
        
    
        if only_I==False:
            loss = torch.mean(torch.abs(pred_y - batch_y))
        else:
            pred_I = pred_y[:,:,1]
            batch_I = batch_y[:,:,1]
            loss = torch.mean(torch.abs(pred_I - batch_I))
        
        return loss.item()
    
    # define a search space
    from hyperopt import hp
    
    space = {}
    space['sigma'] = hp.uniform('sigma', 0.0, 2.0)
    space['mu'] = hp.uniform('mu', 4.0, 6.0)


    # minimize the objective over the space
    from hyperopt import fmin, tpe
    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals)


    print(best)
    # -> {'a': 1, 'c2': 0.01420615366247227}
    print(hyperopt.space_eval(space, best))
    # -> ('case 2', 0.01420615366247227}

    return best




def hyper_min_1(func, func_m, batch_t, batch_y, method, max_evals=100):
    
    # define an objective function
    def objective(args):
        # print(args)
        sigma, mu, beta, gamma = args['sigma'], args['mu'], args['beta'], args['gamma']
        
        func_m.sigma = nn.Parameter(torch.tensor(sigma).to(device), requires_grad=True)
        func_m.mu = nn.Parameter(torch.tensor(mu).to(device), requires_grad=True)
        func.beta = nn.Parameter(torch.tensor(beta).to(device), requires_grad=True)
        func.gamma = nn.Parameter(torch.tensor(gamma).to(device), requires_grad=True)
    
        # idx = np.array([0])
        # batch_y = torch.tensor(data[idx, ...], dtype=torch.float32).to(device)
        batch_y0 = batch_y[:,0,:].to(device)
    
        pred_y = odeint(func, func_m, batch_y0, batch_t, method=method).to(device)
        # pred_y = odeint(func, func_m, batch_y0, batch_t, method='euler').to(device)
        pred_y = pred_y.transpose(1,0)
        
    
        only_I = True        
        if only_I==False:
            loss = torch.mean(torch.abs(pred_y - batch_y))
        else:
            pred_I = pred_y[:,:,1]
            batch_I = batch_y[:,:,1]
            loss = torch.mean(torch.abs(pred_I - batch_I))
        
        return loss.item()
    
    # define a search space
    from hyperopt import hp
    
    space = {}
    space['sigma'] = hp.uniform('sigma', 0.0, 2.0)
    space['mu'] = hp.uniform('mu', 4.0, 6.0)
    space['beta'] = hp.uniform('beta', 1., 3.)
    space['gamma'] = hp.uniform('gamma', 0.5, 1.5)


    # minimize the objective over the space
    from hyperopt import fmin, tpe
    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals)


    print(best)
    # -> {'a': 1, 'c2': 0.01420615366247227}
    print(hyperopt.space_eval(space, best))
    # -> ('case 2', 0.01420615366247227}

    return best





def hyper_min_2(func, func_m, batch_t, batch_y, method, max_evals=100):
    
    # define an objective function
    def objective(args):
        # print(args)
        sigma, mu, beta, gamma, S0 = args['sigma'], args['mu'], args['beta'], args['gamma'], args['S0']
        
        func_m.sigma = nn.Parameter(torch.tensor(sigma).to(device), requires_grad=True)
        func_m.mu = nn.Parameter(torch.tensor(mu).to(device), requires_grad=True)
        func.beta = nn.Parameter(torch.tensor(beta).to(device), requires_grad=True)
        func.gamma = nn.Parameter(torch.tensor(gamma).to(device), requires_grad=True)
        
        I0 = batch_y[:,0,1].to(device)
        func.S0 = nn.Parameter(torch.tensor([S0]).to(device), requires_grad=True)
        R0 = nn.Parameter(torch.tensor([1-func.S0.item()-I0.item()]).to(device), requires_grad=True)
        
        batch_y0 = torch.cat([func.S0,I0,R0]).reshape(1,3)

        # idx = np.array([0])
        # batch_y = torch.tensor(data[idx, ...], dtype=torch.float32).to(device)

        
        pred_y = odeint(func, func_m, batch_y0, batch_t, method=method).to(device)
        # pred_y = odeint(func, func_m, batch_y0, batch_t, method='euler').to(device)
        pred_y = pred_y.transpose(1,0)
        
    
        only_I = True        
        if only_I==False:
            loss = torch.mean(torch.abs(pred_y - batch_y))
        else:
            pred_I = pred_y[:,:,1]
            batch_I = batch_y[:,:,1]
            loss = torch.mean(torch.abs(pred_I - batch_I))
        
        return loss.item()
    
    # define a search space
    from hyperopt import hp
    
    space = {}
    space['sigma'] = hp.uniform('sigma', 0.0, 5.0)
    space['mu'] = hp.uniform('mu', 0.0, 10.0)
    space['beta'] = hp.uniform('beta', .5, 10.)
    space['gamma'] = hp.uniform('gamma', 0.1, 3.5)
    space['S0'] = hp.uniform('S0', 0., 1.)

    # minimize the objective over the space
    from hyperopt import fmin, tpe
    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals)


    print(best)
    # -> {'a': 1, 'c2': 0.01420615366247227}
    print(hyperopt.space_eval(space, best))
    # -> ('case 2', 0.01420615366247227}

    return best




def hyper_min_3(func, func_m, batch_t, batch_y, method, init, max_evals=100):
    
    # define an objective function
    def objective(args):
        # print(args)
        sigma, mu, S0 = args['sigma'], args['mu'], args['S0']
        
        func_m.sigma = nn.Parameter(torch.tensor(sigma).to(device), requires_grad=True)
        func_m.mu = nn.Parameter(torch.tensor(mu).to(device), requires_grad=True)
        # func.beta = nn.Parameter(torch.tensor(beta).to(device), requires_grad=True)
        # func.gamma = nn.Parameter(torch.tensor(gamma).to(device), requires_grad=True)
        
        I0 = batch_y[:,0,1].to(device)
        func.S0 = nn.Parameter(torch.tensor([S0]).to(device), requires_grad=True)
        R0 = nn.Parameter(torch.tensor([1-func.S0.item()-I0.item()]).to(device), requires_grad=True)
        
        batch_y0 = torch.cat([func.S0,I0,R0]).reshape(1,3)

        # idx = np.array([0])
        # batch_y = torch.tensor(data[idx, ...], dtype=torch.float32).to(device)

        
        pred_y = odeint(func, func_m, batch_y0, batch_t, method=method).to(device)
        # pred_y = odeint(func, func_m, batch_y0, batch_t, method='euler').to(device)
        pred_y = pred_y.transpose(1,0)
        
    
        only_I = True        
        if only_I==False:
            loss = torch.mean(torch.abs(pred_y - batch_y))
        else:
            pred_I = pred_y[:,:,1]
            batch_I = batch_y[:,:,1]
            loss = torch.mean(torch.abs(pred_I - batch_I))
        
        return loss.item()
    
    # define a search space
    from hyperopt import hp
    
    space = {}
    space['sigma'] = hp.uniform('sigma', 0.0, 5.0)
    space['mu'] = hp.uniform('mu', 0.0, 10.0)
    # space['beta'] = hp.uniform('beta', .5, 10.)
    # space['gamma'] = hp.uniform('gamma', 0.1, 3.5)
    space['S0'] = hp.uniform('S0', 0., 1.)

    # minimize the objective over the space
    from hyperopt import fmin, tpe
    init_vals = [{'sigma':init[0], 'mu':init[1], 'S0':init[2]}]
    
    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, points_to_evaluate=init_vals)


    print(best)
    # -> {'a': 1, 'c2': 0.01420615366247227}
    print(hyperopt.space_eval(space, best))
    # -> ('case 2', 0.01420615366247227}

    return best





