#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 09:16:53 2023

@author: dliu
"""

import numpy as np
import matplotlib.pyplot as plt

class Point():
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.state = 'S'
        self.t = 0
        
    @property
    def position(self):
        return self.x, self.y
    
    @position.setter
    def position(self, pos):
        self.x = pos[0]
        self.y = pos[1]
        
    @property
    def status(self):
        return self.state
    
    @status.setter
    def status(self, c):
        assert c in ['S', 'I', 'R']
        self.state = c
        
    @property
    def time(self):
        return self.t
    
    @time.setter
    def time(self, t):
        self.t = t

# def p_update(p, h, env, prob=0.6, t1=14, t2=24):
    
#     direct = np.zeros([2], dtype=np.int32)
#     direct[0] = np.random.choice(h*2+1)-h
#     direct[1] = np.random.choice(h*2+1)-h

#     p.position = np.clip(p.position[0]+direct[0], 0, boundary[0]-1), np.clip(p.position[1]+direct[1], 0, boundary[1]-1)
    
#     if p.state=='S' and env=='I':
#         if np.random.rand()<prob:
#             p.state = 'I'
#             p.t = 0
#     elif p.state=='I' and p.t<=t1:
#         p.t += 1
#     elif p.state=='I' and p.t>t1:
#         p.state = 'R'
#         p.t += 1
#     elif p.state=='R' and p.t<=t2:
#         p.t +=1
#     elif p.state=='R' and p.t>t2:
#         p.state = 'S'
#         p.t = 0
        

mean, var = 70, 1
from scipy.stats import norm
def p_update(p, h, env, prob=2.3/14, t1=14, mean=mean, var=var):
    ### gamma=1/t1 in SIR
    ### beta = prob in SIR
    ### R0 = beta/gamma
    
    direct = np.zeros([2], dtype=np.int32)
    direct[0] = np.random.choice(h*2+1)-h
    direct[1] = np.random.choice(h*2+1)-h

    p.position = np.clip(p.position[0]+direct[0], 0, boundary[0]-1), np.clip(p.position[1]+direct[1], 0, boundary[1]-1)
    
    if p.state=='S' and env=='I':
        if np.random.rand()<prob:
            p.state = 'I'
            p.t = 0
    elif p.state=='I' and p.t<=t1:
        p.t += 1
    elif p.state=='I' and p.t>t1:
        p.state = 'R'
        p.t += 1
    
    seed = np.random.rand()
    if p.state=='R' and seed<=norm.cdf(p.t, loc=mean, scale=var):
        p.state = 'S'
        p.t = 0
    elif p.state=='R' and seed>norm.cdf(p.t, loc=mean, scale=var):
        p.t += 1
        

def get_map(points, boundary, group):
    mapp = np.zeros(boundary)  
    result = np.zeros(3)
    for i, p in enumerate(points):
        
        env = 'S'
        for p_ in np.array(points)[group==group[i]]:
            if p_.status=='I':
                env = 'I'
                
        x, y = p.position
        
        if env == 'S':
            mapp[x,y] = .3
        elif env == 'I':
            mapp[x,y] = 1.
        
        if p.state=='S':
            result[0] += 1
        elif p.state=='I':
            result[1] += 1
        elif p.state=='R':
            result[2] += 1
    result = result/len(points)
    
    return mapp, result

# mapp = get_map(points, boundary)
# plt.imshow(mapp)


boundary = 50, 50
N = boundary[0] * boundary[1]
I_init = .01

h = 1
neighbours = [(i, j) for i in range(-h,h+1) for j in range(-h,h+1)]

### initial value of each point
points = []
at_least_one_I = False
for _ in range(N):
    x, y = np.random.randint(boundary[0]), np.random.randint(boundary[1])
    p = Point(x,y)
    
    if np.random.rand()<I_init:
        p.status = 'I'
        at_least_one_I = True
    points.append(p)

if at_least_one_I!=True:
    points[-1].status = 'I'
    
position_dict = ((f'{i},{j}', i*boundary[0]+j) for i in range(boundary[0]) for j in range(boundary[1]))
position_dict = dict(position_dict)


length = 500
mapp = np.zeros([length, *boundary])
results = np.zeros([length, 3])
for tt in range(length):
    
    ### state of each point, and group the overlapped points
    group = np.zeros(N, dtype=np.int32)
    state_list = []
    for i, p in enumerate(points):
        group[i] = position_dict[f'{points[i].position[0]},{points[i].position[1]}']
        state_list.append(p.state)
    
    state_list = np.array(state_list)

    mapp[tt], results[tt] = get_map(points, boundary, group)
    
    ### if any 'I' in each position
    env_list = np.zeros(N, dtype=np.int8) #0 is S, 1 is I
    group_ = np.unique(group)
    for g in group_:
        binary = group==g
        if 'I' in state_list[binary]:
            env_list[binary] = 1
        else:
            env_list[binary] = 0
    
    for i, p in enumerate(points):
        
        if env_list[i]==1:
            env = 'I'
        else:
            env = 'S'
            
        p_update(p, h, env, prob=2.3/14, t1=14, mean=mean, var=var)
        
        if i%1000==0:
            print(i)
    
    print(f'##### {tt} #####')
    

np.save('./data/mapp.npy', mapp)
mapp = np.load('../data/mapp.npy')

# plt.figure(1)
# plt.imshow(mapp[0])

# III = []
# for i in range(length):
#     III.append((mapp[i]==1).sum())

# plt.figure(2)    
# plt.plot(III)


np.save('../data/simulation.npy', results)
results = np.load('../data/simulation_2_3.npy')

plt.figure(3)
plt.plot(results, label=['S', 'I', 'R'])
plt.legend()

