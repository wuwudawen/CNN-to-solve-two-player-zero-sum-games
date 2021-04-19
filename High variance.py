# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 11:59:12 2021

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
    return m.objval, m.Runtime

def uu(game_size):
    l = 75
    games = np.zeros((100, game_size, game_size))
    v = np.zeros((100, ))
    for i in range(100):
        mu, sigma = uniform(0, l), uniform(l, 2*l)
        games[i] = uniform(mu, sigma, (game_size, game_size))
        v[i] = LP_NE(games[i])[0]
    v = pd.DataFrame(data=v)
    print(v.mean())
    print(v.var())
    print(v.skew())
    print(kurtosis(v))
    table(games)
    return v

d = {}
d["10*10"] = uu(10).to_numpy().reshape((-1, ))
d["50*50"] = uu(50).to_numpy().reshape((-1, ))
d["100*100"] = uu(100).to_numpy().reshape((-1, ))
ax = sns.kdeplot(data=d)
ax.set(xlabel="Value", ylabel = "Density")
ax.set_title('UU' + ' ' + 'distribution', fontsize=15)

