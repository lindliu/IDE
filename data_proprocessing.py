#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 12:30:35 2023

@author: do0236li
"""
###https://www.healthdata.org/node/7425
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def figsave(data_, country):
    recovery_time = 10

    fig, ax = plt.subplots(1,5,figsize=[25,5])
    
    ax[0].plot(data_['date'], data_['cases_mean'])
    ax[0].set(xlabel="Date",
           ylabel="reported cases",
           title=f"{country}")
    plt.setp(ax[0].get_xticklabels(), rotation=45)
    
    proportion_cases = np.convolve(recovery_time*[1], data_['cases_mean'], mode='same') / data_['population'].iloc[0]
    ax[1].plot(data_['date'], proportion_cases)
    ax[1].set(xlabel="Date",
           ylabel="reported cases proportion",
           title=f"{country}")
    plt.setp(ax[1].get_xticklabels(), rotation=45)
    
    ax[2].plot(data_['date'], data_['inf_mean'])
    ax[2].set(xlabel="Date",
           ylabel="estimated cases",
           title=f"{country}")
    plt.setp(ax[2].get_xticklabels(), rotation=45)
    
    proportion = np.convolve(recovery_time*[1], data_['inf_mean'], mode='same') / data_['population'].iloc[0]
    ax[3].plot(data_['date'], proportion)
    ax[3].set(xlabel="Date",
           ylabel="proportion",
           title=f"{country}")
    plt.setp(ax[3].get_xticklabels(), rotation=45)
    
    ax[4].plot(data_['date'], data_['reff_mean'])
    ax[4].set(xlabel="Date",
           ylabel="R",
           title="R")
    plt.setp(ax[4].get_xticklabels(), rotation=45)
    
    fig.savefig(f'./data/countries/{country}.png', bbox_inches='tight', pad_inches=0)
    plt.close()



# path_list = ['../data/covid/2022/data_download_file_reference_2020.csv',
#               '../data/covid/2022/data_download_file_reference_2021.csv',
#               '../data/covid/2022/data_download_file_reference_2022.csv']

# datasets = pd.concat([pd.read_csv(path) for path in path_list])

# data = datasets[['date', 'location_name', 'inf_mean', 'cases_mean', 'reff_mean', 'daily_cases', 'population']]
# data['date'] = pd.to_datetime(data['date'])
# data.to_csv('./data/covid_estimated.csv', sep='\t', index=False)

data = pd.read_csv('./data/covid_estimated.csv', sep='\t')
data['date'] = pd.to_datetime(data['date'])

countries = data['location_name'].unique()
for i in range(countries.shape[0]):
    print(i)
    data_ = data[data['location_name']==countries[i]].reset_index()
    if data_['date'].shape[0]==data_['date'].unique().shape[0]:
        figsave(data_, f'{i}:' + countries[i])


countries = data['location_name'].unique()
idx = [16, 24, 34, 43, 49, 51, 54, 68, 92, 110, 133, 135, 138, 139, 228]
for i in idx:
    data_ = data[data['location_name']==countries[i]].reset_index(drop=True)
    
    recovery_time = 10
    proportion = np.convolve(recovery_time*[1], data_['inf_mean'], mode='same') / data_['population'].iloc[0]
    data_.insert(2, "proportion", proportion, True)

    plt.plot(data_['inf_mean'])
    data_.to_csv(f'./data/covid_{countries[i]}.csv', sep='\t', index=False)




