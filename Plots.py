# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 00:10:29 2021

@author: Dawen Wu
"""
import numpy as np
import pandas as pd
import torch
import gurobipy as gp
from gurobipy import GRB
from numpy.random import uniform, normal, poisson
from scipy.stats import kurtosis
from matplotlib.pyplot import plot
import seaborn as sns
sns.set_theme(style="whitegrid")

distributions = {'Uniform': uniform, 'Normal': normal, 'Poisson': poisson}
distri_param = {'Uniform': (-10, 100), 'Normal': (25, 30), 'Poisson': (35, )}
game_sizes = [10, 50, 100]
n_sample = 100

def NN_loss(iterations):
    nngeneral_record = torch.load(r'MCNN_record.pt', map_location=torch.device('cpu'))
    x = torch.load(r'MCNN_record.pt', map_location=torch.device('cpu')).numpy()
    x = x[:iterations]
    sns.set_theme(style="whitegrid")
    ax = sns.lineplot(data=x)
    ax.set(xlabel="Iterations", ylabel = "MSE")
    ax.set_title('Training loss')
    
def LP_NE(rewardM):
    m = gp.Model()
    m.setParam('OutputFlag', 0)

    game_size = rewardM.shape[0]
    x = m.addMVar(shape=game_size+1, lb=[0]*game_size+[-float('inf')], ub=[1]*game_size+[float('inf')])
    
    c = np.array([0]*game_size + [1])
    m.setObjective(c @ x, GRB.MAXIMIZE)
    A_1 = np.concatenate((-rewardM, np.ones((game_size,1))), axis=1)
    b_1 = np.zeros(game_size)
    m.addConstr(A_1 @ x <= b_1)
    A_2 = np.array([1]*game_size+[0])
    b_2 = 1
    m.addConstr(A_2 @ x == 1)
    
    m.optimize()
    return m.objval

# Figure 4
def figure(distibution):
    d = {}
    for game_size in game_sizes:
        res = np.zeros((n_sample, ))
        for i in range(n_sample):
            distri = distributions[distibution]
            p = distri_param[distibution]
            rewardM = distri(*p, (game_size, game_size))
            res[i] = LP_NE(rewardM)
        d[f'{game_size}*{game_size} '] = res.copy()
    
    df = pd.DataFrame(data=d)
    ax = sns.kdeplot(data=df)
    ax.set(xlabel="Value", ylabel = "Density")
    ax.set_title(distibution + ' ' + 'distribution', fontsize=15)
    return df

# Table 1
dfU = figure('Uniform')
dfN = figure('Normal')
dfP = figure('Poisson')

dfU1 = dfU.mean()
dfN1 = dfN.mean()
dfP1 = dfP.mean()

dfU2 = dfU.var()
dfN2 = dfN.var()
dfP2 = dfP.var()

dfU3 = dfU.skew()
dfN3 = dfN.skew()
dfP3 = dfP.skew()

dfU4 = kurtosis(dfU, fisher=False)
dfN4 = kurtosis(dfN, fisher=False)
dfP4 = kurtosis(dfP, fisher=False)

