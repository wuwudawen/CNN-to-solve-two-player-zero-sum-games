#Neccesary packages
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.random import uniform
from numpy.random import normal
from numpy.random import poisson
import gurobipy as gp
from gurobipy import GRB
import torch.nn.functional as F

#Fixed Parmeters1
epoch = 1000000000
criterion = nn.MSELoss()
record = []
cuda = torch.cuda.is_available()
lr = 0.00001

#Tunable Parmeters1: batch_size and train_round
batch_size = 60
train_round = 10

#Tunable Parmeters2: game sizes and generating distributions
game_sizes = [10, 30, 50, 70, 100, 200]
def distributions(game_size):
    mb = int(batch_size/3)
    X_1 = uniform(-10, 100, (mb, 1, game_size, game_size))
    X_2 = normal(25, 3, (mb, 1, game_size, game_size))
    X_3 = poisson(35, (mb, 1, game_size, game_size))
    X = np.concatenate((X_1, X_2, X_3),0)
    return X

#Neural network structure
class GNet(nn.Module):
    def __init__(self):
        super(GNet, self).__init__()
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

#Linear programming solving matrix games
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

#Generate a batch of data (X, Y)
def create_data(game_size):
    X = distributions(game_size)
    Y = np.zeros((batch_size, 1))
    for i in range(batch_size):
        Y[i, 0] = LP_NE(X[i, 0, :, :])
    X = torch.tensor(X).view(batch_size, 1, game_size, game_size).float()
    Y = torch.tensor(Y).view(batch_size, 1).float()
    if cuda: 
        X, Y = X.cuda(), Y.cuda()
    return (X, Y)

#Train the CNN model
def train(net):
    optimizer = optim.Adam(net.parameters(), lr=lr)
    for i in range(epoch):
        for game_size in game_sizes:
            X, Y = create_data(game_size)
            for j in range(train_round):
                optimizer.zero_grad()
                output = net(X)
                loss = criterion(output, Y)
                loss.backward()
                optimizer.step()
                record.append(loss)
            print(f'The {i}th data, {game_size}*{game_size} game size, loss: {loss}')

net = GNet()
if cuda:
    net = net.cuda()

