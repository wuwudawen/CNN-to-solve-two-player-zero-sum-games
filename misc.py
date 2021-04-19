# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:42:52 2021

@author: Dawen Wu
"""

def scipy_LP(rewardM):
    def LP_model(rewardM):
        m,n = rewardM.shape[0], rewardM.shape[1]
        
        firstcolumn, left = -rewardM.T[:,[0]],-rewardM.T[:,1:]
        matrix = -firstcolumn+left
        a,b = -np.ones((1,m)), np.ones((1,m))
        a[0,-1],b[0,-1] = 0,0
        A_ieq_1 = np.concatenate((matrix,np.ones((n,1))), axis=1)
        A_ieq_2 = np.concatenate((np.identity(m-1),np.zeros((m-1,1))),axis=1)
        A_ieq_2 = np.concatenate((a,A_ieq_2),axis=0)
        A_ieq_3 = np.concatenate((-np.identity(m-1),np.zeros((m-1,1))),axis=1)
        A_ieq_3 = np.concatenate((b,A_ieq_3),axis=0)
        
        c = [0]*(m-1) + [-1]
        A_ieq = np.concatenate((A_ieq_1,A_ieq_2,A_ieq_3),axis=0)
        b_ieq = np.concatenate((-firstcolumn.reshape((-1,)),[0]+[1]*(m-1),[1]+[0]*(m-1)),axis = 0)
        return c, A_ieq, b_ieq

    m,n = rewardM.shape[0], rewardM.shape[1]
    c, A_ieq, b_ieq = LP_model(rewardM)
    
    start = time.time()
    res = linprog(c, A_ub=A_ieq, b_ub=b_ieq, bounds=[(None,None)]*m)
    CPU_time = time.time() - start
    value = -res.fun
    
    print(f"Linear programming: \nValue: {value:.4f}, time: {CPU_time:.4f} ")
    return value

def scipy_LS_y(rewardM, value):
    m = rewardM.shape[0]
    c = np.zeros((m,))
    A_ub = rewardM
    b_ub = np.ones((m, ))*value
    A_eq = np.ones((1, m))
    b_eq = np.ones((1, ))
    
    start = time.time()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0,1)]*m)
    CPU_time = time.time() - start
    print(res.success)
    return CPU_time


def findx(rewardM, value, y):
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    n, m = rewardM.shape[0], rewardM.shape[1]
    
    x = model.addMVar(shape=m, lb=[0]*n, ub=[1]*n)
    
    model.setObjective(0, GRB.MINIMIZE)
    vec = (rewardM@y).reshape((100,))
    model.addConstr(x@vec == value)
    model.addConstr(np.ones((1,m))@x == 1)
    model.optimize()
    return np.array(model.x)
