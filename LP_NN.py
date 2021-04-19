# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 18:02:40 2021

@author: 40691
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import uniform, normal, poisson
import gurobipy as gp
from gurobipy import GRB
import time

# A CNN structure capable to receive arbitary input
class Generalnet(nn.Module):
    def __init__(self):
        super(Generalnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 1)
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.max_pool2d(F.leaky_relu(self.conv2(x)), 2)
        x = F.leaky_relu(self.conv3(x))
        x = F.max_pool2d(F.leaky_relu(self.conv4(x)), 2)
        x = x.mean(dim=(-2, -1))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
# Load CNN model
mcnn = Generalnet()
mcnn.load_state_dict(torch.load('MCNN_model.pt', map_location=torch.device('cpu')))

# The LP method to solve matrix games
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
    
    value = m.objval
    CPU_time = m.Runtime
    return value, CPU_time

# The CNN method to solve matrix games
def CNN_NE(rewardM):
    game_size = rewardM.shape[1]
    rewardM = torch.tensor(rewardM).float().view(-1, 1, game_size, game_size)
    start = time.time()
    value = mcnn(rewardM)
    CPU_time = time.time() - start
    return value, CPU_time          
            
# The result, accuracy and computational time
def table(rewardM):
    n_sample = rewardM.shape[0]
    game_size = rewardM.shape[1]
    
    LP_record = np.zeros((n_sample, 2), dtype=float)
    NN_record = np.zeros((n_sample, 2), dtype=float)
    for i in range(n_sample):
        LP_record[i, 0], LP_record[i, 1] = LP_NE(rewardM[i])
        NN_record[i, 0], NN_record[i, 1] = CNN_NE(rewardM[i])

    LP_mv, LP_mt = LP_record[:, 0].mean(), LP_record[:, 1].mean()
    NN_mv, NN_mt = NN_record[:, 0].mean(), NN_record[:, 1].mean()
    difference_NN = np.abs(LP_record[:, 0] - NN_record[:, 0]).mean()
    gap_NN = np.abs((LP_record[:, 0] - NN_record[:, 0])/NN_record[:, 0]).mean()

    print(f"** number sample: {n_sample}, game size: {game_size} **\n")
    print(f"Linear programming         : mean time: {LP_mt:.4f}, mean value: {LP_mv:.4f} ")
    print(f"Convolutinal Neural Network: mean time: {NN_mt:.4f}, mean value: {NN_mv:.4f},  mean gap: {gap_NN*100:.2f}%")

rewardM = uniform(-10, 100, (100, 35, 35)) + normal(25, 3, (100, 35, 35)) + poisson(35, (100, 35, 35))
table(rewardM)