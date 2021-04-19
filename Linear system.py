from scipy.optimize import linprog

def Linearsystem_x(rewardM, value):
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    n, m = rewardM.shape[0], rewardM.shape[1]

    x = model.addMVar(shape=n, lb=[0]*n, ub=[1]*n)
    
    model.setObjective(0, GRB.MAXIMIZE)
    model.addConstr(rewardM.T@x >= np.ones((m, ))*value)
    model.addConstr(np.ones((1,n))@x == 1)
    model.optimize()
    
    if model.status == 2:
        return np.array(model.x)
    else:
        return False
    
def Linearsystem_y(rewardM, value):
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    n, m = rewardM.shape[0], rewardM.shape[1]
    
    y = model.addMVar(shape=m, lb=[0]*m, ub=[1]*m)
    
    model.setObjective(0, GRB.MINIMIZE)
    model.addConstr(rewardM@y <= np.ones((n, ))*value)
    model.addConstr(np.ones((1,m))@y == 1)
    model.optimize()
    
    if model.status == 2:
        return np.array(model.x)
    else:
        return False

def LP_x(rewardM):
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    n, m = rewardM.shape[0], rewardM.shape[1]

    x = model.addMVar(shape=n, lb=[0]*n, ub=[1]*n)
    v = model.addMVar(shape=1, lb=-float('inf'), ub=float('inf'))
    
    model.setObjective(v, GRB.MAXIMIZE)
    model.addConstr(rewardM.T@x >= np.ones((m, 1))@v)
    model.addConstr(np.ones((1,n))@x == 1)
    model.optimize()
    return model.objval, model.Runtime

def LP_y(rewardM):
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    n, m = rewardM.shape[0], rewardM.shape[1]
    
    y = model.addMVar(shape=m, lb=[0]*m, ub=[1]*m)
    v = model.addMVar(shape=1, lb=-float('inf'), ub=float('inf'))
    
    model.setObjective(v, GRB.MINIMIZE)
    model.addConstr(rewardM@y <= np.ones((n, 1))@v)
    model.addConstr(np.ones((1,m))@y == 1)
    model.optimize()
    return model.objval, model.Runtime

def LS_xy(rewardM, value):
    game_size = rewardM.shape[1]

    x = Linearsystem_x(rewardM, value)
    if x is not False:
        y = np.zeros((game_size,))
        index = np.argmin(x.T@rewardM)
        y[index] = 1
        return x, y
    else:
        y = Linearsystem_y(rewardM, value)
        x = np.zeros((game_size,))
        index = np.argmax(rewardM@y)
        x[index] = 1
        return x, y


rewardM = uniform(-10,100,(100, 100))
value = CNN_NE(rewardM)[0].item()
x, y = LS_xy(rewardM, value)
