#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 13:44:05 2022

@author: dliu
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dist = np.load('./data/dist.npy')
dist = torch.tensor(dist, dtype=torch.float32).to(device)

beta = 1.5 #.5
gamma = 1 #.1

class funcI(nn.Module):    
    def __init__(self):
        super(funcI, self).__init__()

        self.memory = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1),
            # nn.ReLU()
            nn.Sigmoid()
        )
    def forward(self, t):
        
        # t = torch.flip(t, [0]).reshape([-1,1])
        # t = t.repeat(1,2).reshape(-1,1)
        return self.memory(t)
    

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
    def forward(self, t):
        
        # t = torch.flip(t, [0]).reshape([-1,1])
        # t = t.repeat(1,2).reshape(-1,1)
        return self.memory(t)



# fff = Memory().to(device)
# optimizer = optim.RMSprop(fff.parameters(), lr=1e-3)

# batch_y0 = batch_y[:,[0],:].to(device)
# for itr in range(1, 5000):
#     optimizer.zero_grad()
    
#     pred_dist = fff(batch_t)
#     loss = nn.functional.mse_loss(pred_dist, torch.tensor(dist[-40:],dtype=torch.float32).to(device))

#     loss.backward()
#     optimizer.step()
    
#     if itr%100==0:
#         print(loss.item())

# pred_dist = fff(batch_t)  
# plt.plot(pred_dist.cpu().detach())
# plt.plot(dist[-40:].cpu())


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        # for m in self.memory.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, mean=0, std=0.1)
        #         nn.init.constant_(m.bias, val=0)
                
        
        self.g = Memory().to(device)#(time)
        # self.g = fff
        
        self.funcI = funcI().to(device)
        
        
        # self.beta = 1.5#nn.Parameter(torch.tensor(0.1).to(device), requires_grad=True)
        # self.gamma = 1#nn.Parameter(torch.tensor(0.1).to(device), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(0.2).to(device), requires_grad=True)  ## initial value matters, if we choose 1.5 then it fails
        self.gamma = nn.Parameter(torch.tensor(0.2).to(device), requires_grad=True)
        
        # self.dx = 0.40578396351620016
    def forward(self, t, y):
        self.dt = t[0]-t[1]
        me = self.g(t)
        
        # me = dist
        
        self.me = me
        
        # series = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)        
        
        y = y.transpose(1,0)
        solution = y.clone()
        # print(solution.shape)
        
        diff = torch.zeros(t.shape[0], y.shape[1], 3, dtype=torch.float32).to(device)

        for ii in range(1,t.shape[0]):
            diff[ii-1,:,:] = self.ff(solution, me, ii)[0]
            y = y + self.ff(solution, me, ii)*self.dt
            solution = torch.cat((solution, y), dim=0)


        return solution, diff
        # return torch.cat((dSdt,dIdt,dRdt),1)
    
    def ff(self, solution, me, j):
        self.integro = self.integral(solution,me,j)
        # print('dd', self.integro.shape)

        beta = self.beta
        gamma = self.gamma
        
        S, I, R = torch.split(solution[[-1],:,:],1,dim=2)
        # print(S.shape)

        dSdt = -beta * S * I + self.integro
        dIdt = beta * S * I - gamma * I
        # dIdt = self.funcI(torch.cat((S,R),2).transpose(1,0)).transpose(1,0)
        dRdt = gamma * I - self.integro
        
        # print('ddfsf', dRdt.shape, dIdt.shape)
        return torch.cat((dSdt,dIdt,dRdt),2)
    
    def integral(self, solution, me, j):
        S, I, R = torch.split(solution,1,dim=2)

        batchsize = S.shape[1]
                
        integro = I[:,:,0] * me[-j:] * self.dt
        # print(integro.sum(0))

        return integro.sum(0).reshape(1, batchsize, 1)
    

k = 5
t = torch.linspace(0., 80./k, 200//k).to(device)
t = torch.flip(t, [0]).reshape([-1,1])
# t = t.repeat(1,2).reshape(-1,1)

func = ODEFunc().to(device)
y0 = torch.tensor([[[0.9696, 0.0215, 0.0090]], [[0.66667574, 0.09769345, 0.23663081]]], dtype=torch.float32).to(device)
results, diff = func(t, y0)
results = results.transpose(1,0)
diff = diff.transpose(1,0)

plt.figure()
plt.plot(results[0,:,:].cpu().detach())




import numpy as np
traindata = np.load('./data/train_sir.npy')






'''
###################################
############## RMSprop ############
###################################

k = 5
t = torch.linspace(0., 80./k, 200//k).to(device)
t = torch.flip(t, [0]).reshape([-1,1])

# batch = traindata[np.arange(batch_size).reshape(-1,1)+np.arange(200//k)]
batch_y = torch.tensor(traindata[:,:200//k,:], dtype=torch.float32).to(device)
batch_y0 = batch_y[:,[0],:].to(device)
batch_t = t
diff_num = (batch_y[:,1:,:] - batch_y[:,:-1,:])/(t[1]-t[0])

func = ODEFunc().to(device)
optimizer = optim.Adam(func.parameters(), lr=1e-4)
for itr in range(1, 10000):
    optimizer.zero_grad()
    
    pred_y, diff = func(batch_t, batch_y0)
    pred_y = pred_y.transpose(1,0)
    diff = diff.transpose(1,0)

    loss_mse = torch.mean(torch.abs(pred_y - batch_y))
    loss_diff = torch.mean(torch.abs(diff[:,:-1,:] - diff_num))
    # loss = loss_mse
    loss = loss_diff+loss_mse

    loss.backward()
    optimizer.step()
    
    if itr%100==0:
        print(f'itr: {itr}. loss_diff: {loss_diff.item():.5e}, loss_mse: {loss_mse.item():.5e}')

'''


#################################
############## LBFGS ############
#################################
k = 5
t = torch.linspace(0., 80./k, 200//k).to(device)
t = torch.flip(t, [0]).reshape([-1,1])

batch_y = torch.tensor(traindata[:,:200//k,:], dtype=torch.float32).to(device)
batch_y0 = batch_y[:,[0],:].to(device)
batch_t = t
diff_num = (batch_y[:,1:,:] - batch_y[:,:-1,:])/(t[1]-t[0])

func = ODEFunc().to(device)
optimizer = optim.LBFGS(func.parameters(), lr=5e-2)
for itr in range(1,100):
    def closure():
        optimizer.zero_grad()
        
        pred_y, diff = func(batch_t, batch_y0)
        pred_y = pred_y.transpose(1,0)
        diff = diff.transpose(1,0)[:,:-1,:]
    
        loss_mse = nn.functional.mse_loss(pred_y, batch_y)
        loss_diff = nn.functional.mse_loss(diff, diff_num)
        loss_sum = nn.functional.mse_loss(pred_y.sum(2), torch.ones_like(pred_y.sum(2)))
        
        loss = loss_mse
        # loss = loss_diff + loss_mse# + .01*loss_sum
        print(f'loss_diff: {loss_diff.item():.3e}, loss_mse: {loss_mse.item():.3e}, beta:{func.beta.item():.2f}, gamma:{func.gamma.item():.2f}')
        loss.backward()
        return loss
    optimizer.step(closure)
    
#################################
#################################




pred_y, diff = func(batch_t, batch_y0)
pred_y = pred_y.transpose(1,0)
diff = diff.transpose(1,0)


jj = 20
fig, ax = plt.subplots(1,2,figsize=[12,4])
ax[0].plot(pred_y[jj,:,:].cpu().detach(), label=['predict S', 'predict I', 'predict R'])
ax[0].plot(batch_y[jj,:,:].cpu(), label=['S', 'I', 'R'])
ax[0].plot([],'.',label=f'MSE: {((batch_y-pred_y)**2).mean().item():.2e}')
ax[0].plot([],'.',label=f'beta(1.5): {func.beta.item():.2f}')
ax[0].plot([],'.',label=f'gamma(1): {func.gamma.item():.2f}')
ax[0].legend()

ax[1].plot(func.me.cpu().detach()*func.dt.cpu(), label='predict dist')
ax[1].plot(dist[-40:].cpu(), label='dist')
ax[1].plot([],'.',label=f'MSE: {((func.me.cpu().detach()-dist[-40:].cpu())**2).mean().item():.2e}')
ax[1].legend()
# fig.savefig('./figures/unkonwn_I_without_diff.png')
# fig.savefig('./figures/unkonwn_I_with_diff.png')
# fig.savefig('./figures/konwn_I_without_diff.png')
# fig.savefig('./figures/konwn_I_with_diff.png')

plt.figure()
plt.plot(diff[jj].cpu().detach())
plt.plot(diff_num[jj].cpu())







testdata = np.load('./data/test_sir.npy')

test_y = torch.tensor(testdata[:,:200//k,:], dtype=torch.float32).to(device)
test_y0 = test_y[:,[0],:].to(device)

pred_y, diff = func(batch_t, test_y0)
pred_y = pred_y.transpose(1,0)
diff = diff.transpose(1,0)


jj = 20
fig, ax = plt.subplots(1,2,figsize=[12,4])
ax[0].plot(pred_y[jj,:,:].cpu().detach(), label=['predict S', 'predict I', 'predict R'])
ax[0].plot(test_y[jj,:,:].cpu(), label=['S', 'I', 'R'])
ax[0].plot([],'.',label=f'MSE: {((test_y-pred_y)**2).mean().item():.2e}')
ax[0].plot([],'.',label=f'beta(1.5): {func.beta.item():.2f}')
ax[0].plot([],'.',label=f'gamma(1): {func.gamma.item():.2f}')
ax[0].legend()

ax[1].plot(func.me.cpu().detach()*func.dt.cpu(), label='predict dist')
ax[1].plot(dist[-40:].cpu(), label='dist')
ax[1].plot([],'.',label=f'MSE: {((func.me.cpu().detach()-dist[-40:].cpu())**2).mean().item():.2e}')
ax[1].legend()

plt.figure()
plt.plot(diff[jj].cpu().detach())
plt.plot(diff_num[jj].cpu())

print(f'MSE: {((pred_y-test_y)**2).mean():.2e}') #2.724662408581935e-05 with loss_diff
                                             #1.2362557981759892e-06 without loss_diff



# k = 4
# t = torch.linspace(0., 80./k, 200//k).to(device)
# t = torch.flip(t, [0]).reshape([-1,1])

# y0 = torch.tensor([[[1, 0.001, 0]], [[1.,0.002,0.]]], dtype=torch.float32).to(device)
# results, diff = func(t, y0)
# results = results.transpose(1,0)
# diff = diff.transpose(1,0)

# plt.figure()
# plt.plot(results[0,:,:].cpu().detach())


