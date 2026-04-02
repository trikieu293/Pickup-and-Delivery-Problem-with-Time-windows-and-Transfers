import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time

import gurobipy as gp
from gurobipy import GRB

filename = "./PDPT/PDPT-R7-K3-T3-Q100-6.txt"

# Read the meta-data of problem (number of requests, number of vehicles, number of transport stations, capability of vehicles)
def readMetaData(filename):
    metaData = pd.read_csv(filename, nrows=2, sep= '\t', on_bad_lines='skip')
    return metaData

# Read the instance's data (name of node, location (x, y), time-windows, load of the request)
def readDataframe(filename):
    df = pd.read_csv(filename, skiprows=3, sep='\t')
    return df

# Calculate the Euclid-distance between locations
def calculateDistance(x1, x2, y1, y2):
    return math.sqrt((x2 - x1)**2 + (y2-y1)**2)

# Generate distance-matrix between locations
def distancesMatrix(df):
    matrix = {}
    for location1 in df["node"]:
        for location2 in df["node"]:
            if location1 != location2:
                x1 = df.loc[df["node"]==location1, 'x']
                x2 = df.loc[df["node"]==location2, 'x']
                y1 = df.loc[df["node"]==location1, 'y']
                y2 = df.loc[df["node"]==location2, 'y']
                matrix[location1, location2] = calculateDistance(int(x1), int(x2), int(y1), int(y2))
    return matrix

# Generate dictionary of (node:load)
def loadDict(df):
    matrix = {}
    for location in df["node"]:
        matrix[location] = df.loc[df["node"]==location, 'load'].values[0]
    return matrix

# Get the list of grouped nodes
def getNodeList(df):
    rOrigins = df.loc[df['node'].str.contains('p'),'node']
    rDestinations = df.loc[df['node'].str.contains('d'),'node']
    vOrigins = df.loc[df['node'].str.contains('o'),'node']
    vDestinations = df.loc[df['node'].str.contains('e'),'node']
    transferNodes = df.loc[df['node'].str.contains('t'),'node']
    return [rOrigins, rDestinations, vOrigins, vDestinations, transferNodes]
def calculateDistance(x1, x2, y1, y2):
    return math.sqrt((x2 - x1)**2 + (y2-y1)**2)

def distancesMatrix(df):
    matrix = {}
    for location1 in df["node"]:
        matrix[location1] = {}
        for location2 in df["node"]:
            if location1 != location2:
                x1 = df.loc[df["node"]==location1, 'x'].values[0]
                x2 = df.loc[df["node"]==location2, 'x'].values[0]
                y1 = df.loc[df["node"]==location1, 'y'].values[0]
                y2 = df.loc[df["node"]==location2, 'y'].values[0]
                matrix[location1][location2] = calculateDistance(int(x1), int(x2), int(y1), int(y2))
    return matrix

def loadMatrix(df):
    matrix = {}
    for location in df["node"]:
        matrix[location] = df.loc[df["node"]==location, 'load'].values[0]
    return matrix

def getNodeList(df):
    allNodes = df['node']
    rOrigins = df.loc[df['node'].str.contains('p'),'node']
    rDestinations = df.loc[df['node'].str.contains('d'),'node']
    vOrigins = df.loc[df['node'].str.contains('o'),'node']
    vDestinations = df.loc[df['node'].str.contains('e'),'node']
    transferNodes = df.loc[df['node'].str.contains('t'),'node']
    return {"a":allNodes, "ro":rOrigins, "rd":rDestinations, "vo":vOrigins, "vd":vDestinations, "t":transferNodes}

# Callback gap vs time
def data_cb(model, where):
    if where == gp.GRB.Callback.MIP:
        cur_obj = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
        cur_bd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
        gap = (abs(cur_bd - cur_obj)/abs(cur_obj))*100
        
        # Change in obj value or bound?
        if model._obj != cur_obj or model._bd != cur_bd:
            model._obj = cur_obj
            model._bd = cur_bd
            model._gap = gap
            model._data.append([time.time() - model._start, cur_obj, cur_bd, gap])

# Model, Parameters, Variables and Objective Function
def raisModel(filename):
    model = gp.Model()
    metaData = readMetaData(filename)
    df = readDataframe(filename)
    nodeList = getNodeList(df)
    
    # Testing Symmetries Breaking Constraints
    # df.loc[df['node'].str.contains('o'), 'x'] = 50
    # df.loc[df['node'].str.contains('o'), 'y'] = 50
    # df.loc[df['node'].str.contains('e'), 'x'] = 50
    # df.loc[df['node'].str.contains('e'), 'y'] = 50

    nRequests = int(metaData['nr'])
    nVehicles = int(metaData['nv'])
    nTransports = int(metaData['nt'])
    vCapability = int(metaData['capacity'])

    c = pd.DataFrame.from_dict(distancesMatrix(df)).fillna(0)
    k = pd.RangeIndex(nVehicles)
    r = pd.RangeIndex(nRequests)
    u = pd.Series(index=k, data=np.full(nVehicles, vCapability))
    q = pd.Series(index = np.concatenate((nodeList['ro'].values, nodeList['rd'].values)), data=df.loc[0:nRequests*2-1,'load'].values, dtype=int)

    print(df)

    xIndex = [(k, i, j) for k in pd.RangeIndex(nVehicles) for i in nodeList['a'].values for j in nodeList['a'].values if i != j]
    yIndex = [(k, r, i, j) for k in pd.RangeIndex(nVehicles) for r in pd.RangeIndex(nRequests) for i in nodeList['a'].values for j in nodeList['a'].values if i != j]
    sIndex = [(k1, k2, t, r) for k1 in pd.RangeIndex(nVehicles) for k2 in pd.RangeIndex(nVehicles) for t in nodeList['t'].values for r in pd.RangeIndex(nRequests) if k1 != k2]
    aIndex = [(k, i) for k in pd.RangeIndex(nVehicles) for i in nodeList['a'].values]
    bIndex = [(k, i) for k in pd.RangeIndex(nVehicles) for i in nodeList['a'].values]

    x = model.addVars(xIndex, vtype=GRB.BINARY, name='x')
    y = model.addVars(yIndex, vtype=GRB.BINARY, name='y')
    s = model.addVars(sIndex, vtype=GRB.BINARY, name='s')
    a = model.addVars(aIndex, vtype=GRB.INTEGER, name='a')
    b = model.addVars(bIndex, vtype=GRB.INTEGER, name='b')

    model.setObjective(sum((c[i][j] * x[k, i, j]) for i in nodeList['a'].values for j in nodeList['a'].values for k in pd.RangeIndex(nVehicles) if i != j), GRB.MINIMIZE)
    model.update()


    # Constraints
    for k in pd.RangeIndex(nVehicles):
        for i in nodeList['vo'].values:
            if i.replace('o','') == str(k):
                model.addConstr(sum(x[k, i, j] for j in nodeList['a'].values if j != i) == 1, name="constr1")
                for l in nodeList['vd'].values:
                    if l.replace('e','') == str(k):
                        model.addConstr(sum(x[k, i, j] for j in nodeList['a'].values if j != i) == sum(x[k, j, l] for j in nodeList['a'].values if j != l), name="constr2")
        for i in nodeList['a'].values:
            if not (i.replace('o','') == str(k) or i.replace('e','') == str(k)):
                model.addConstr(sum(x[k, i, j] for j in nodeList['a'].values if i != j) == sum(x[k, j, i] for j in nodeList['a'].values if i != j), name="constr3")

    # Constaints (4), (5), (6), (16Lyu) are used to maintain the request flow              
    for r in pd.RangeIndex(nRequests):
        for i in nodeList['a'].values:
            if i in nodeList['ro'].values:
                if 'p' + str(r) == i:
                    # only one vehicle can pickup the request at request's origin
                    model.addConstr(sum(sum(y[k, r, i, j] for j in nodeList['a'].values if i != j) for k in pd.RangeIndex(nVehicles)) == 1, name="constr4")
                    
            if i in nodeList['rd'].values:
                if 'd' + str(r) == i:
                    # only one vehicle can drop off the request at request's destination
                    model.addConstr(sum(sum(y[k, r, j, i] for j in nodeList['a'].values if i != j) for k in pd.RangeIndex(nVehicles)) == 1, name="constr5")
                    
            if i in nodeList['t'].values:
                # the total number of request go in and out at a specific transfer node must be equal
                model.addConstr(sum(sum(y[k, r, i, j] for j in nodeList['a'].values if i != j) for k in pd.RangeIndex(nVehicles)) == sum(sum(y[k, r, j, i] for j in nodeList['a'].values if i != j) for k in pd.RangeIndex(nVehicles)), name="constr6")   
            
            #Lyu
            for k in pd.RangeIndex(nVehicles):
                if i not in [node for node in nodeList['ro'].values if 'p' + str(r) == node] and i not in [node for node in nodeList['rd'].values if 'd' + str(r) == node] and i not in nodeList['t'].values:
                    # request must be delivered by a same vehicle when it go throught a node other than its origin or destination or a transfer node
                    model.addConstr(sum(y[k, r, i, j] for j in nodeList['a'].values if i != j) == sum(y[k, r, j, i] for j in nodeList['a'].values if i != j), name="constr16")

    for i in nodeList['a'].values:
        for j in nodeList['a'].values:
            for k in pd.RangeIndex(nVehicles):
                if i != j:
                    for r in pd.RangeIndex(nRequests):
                        # synchronisation between vehicle flow and request flow
                        model.addConstr(y[k, r, i, j] <= x[k, i, j], name="constr8")
                        
                    # capacity constraint    
                    model.addConstr(sum((q[r]*y[k, r, i, j]) for r in pd.RangeIndex(nRequests)) <= u[k]*x[k, i, j], name="constr9")
    # # Revisited from (10)-(12) to eliminate the subtours for PDPT                
    # for i in nodeList['a'].values:
    #     for j in nodeList['a'].values:
    #         for k in pd.RangeIndex(nVehicles):
    #             if i != j:                
    #                 model.addConstr(x[k, i, j] <= z[k, i, j], name="constr17")
    #                 model.addConstr(z[k, i, j] + z[k, j, i] == 1, name="constr18")
    #                 for l in nodeList['a'].values:
    #                     if (j != l and i != l):
    #                         model.addConstr(z[k, i, j] + z[k, j, l] + z[k, l, i] <= 2, name="constr19")
                            

    # ensure that the requests are picked up and delivered in the given time windows
    for i in nodeList['a'].values:
        for k in pd.RangeIndex(nVehicles):
            for j in nodeList['a'].values:
                if i != j:
                    M = max(0, df.loc[df['node'] == i, 'b'].values[0]) + c[i][j] - int(df.loc[df['node'] == j, 'a'].values[0])
                    model.addConstr(b[k, i] + c[i][j] - a[k, j] <= M * (1 - x[k, i, j]), name='constr15')
                    
            model.addConstr(a[k, i] <= b[k, i], name='constr16')   
            
            # Constrains (17) and (18) can be combined
            if i in nodeList['ro'].values:
                model.addConstr(a[k, i] >= int(df.loc[df['node'] == i, 'a'].values[0]), name='constr17')
                model.addConstr(b[k, i] <= int(df.loc[df['node'] == i, 'b'].values[0]), name='constr17')
            if i in nodeList['rd'].values:
                model.addConstr(a[k, i] >= int(df.loc[df['node'] == i, 'a'].values[0]), name='constr18')
                model.addConstr(b[k, i] <= int(df.loc[df['node'] == i, 'b'].values[0]), name='constr18')   


    # maintain the synchronisation at transfer nodes
    for r in pd.RangeIndex(nRequests):
        for t in nodeList['t'].values:
            M = int(df.loc[df['node'] == t, 'b'].values[0]) - int(df.loc[df['node'] == t, 'a'].values[0])
            for k1 in pd.RangeIndex(nVehicles):
                for k2 in pd.RangeIndex(nVehicles):
                    if k1 != k2:
                        # maintain the synchronisation variables at transfer nodes
                        model.addConstr(sum(y[k1, r, j, t] for j in nodeList['a'].values if j != t) + sum(y[k2, r, t, j] for j in nodeList['a'].values if j != t) <= (s[k1, k2, t, r] + 1), name='constr19')
                        
                        # maintain the synchronisation time-window at transfer nodes
                        model.addConstr(a[k1, t] - b[k2, t] <= M * (1 - s[k1, k2, t, r]), name='constr20')
        
    # Symmetries Breaking Constraints form Cortes (2010)                
    # for r in pd.RangeIndex(nRequests):
    #     model.addConstr(sum(sum(x[k, 'o' + str(r), j] for j in nodeList['a'].values if 'o' + str(r) != j) for k in pd.RangeIndex(nVehicles) if k > r) == 0, name='symmetries breaking constr')


    # Data for callback
    model._obj = None
    model._bd = None
    model._gap = None
    model._data = []
    model._start = time.time()
    model.Params.TimeLimit = 60*60
    model.update()
    model.optimize(callback=data_cb)
    # model.optimize()
    # model.computeIIS()
    # model.write("model.ilp")

    def plotGap(data):
        dfResult = pd.DataFrame(data, columns=['time', 'cur_obj','cur_bd','gap'])
        dfResult = dfResult.drop(dfResult[dfResult.cur_obj >= 100000000].index)
        
        fig, axes = plt.subplots()
        
        axes.set_xlabel('time')
        axes.set_ylabel('value')
        axes.set_xlim(dfResult['time'].values.min(), dfResult['time'].values.max())
        axes.set_ylim(0, dfResult['cur_obj'].values.max() * 1.1)
        line1, = axes.plot(dfResult['time'].values, dfResult['cur_obj'].values, color = 'navy', label='Current ObjValue')    
        line2, = axes.plot(dfResult['time'].values, dfResult['cur_bd'].values, color = 'blue', label='Current DB')    
        plt.fill_between(dfResult['time'].values, dfResult['cur_obj'].values, dfResult['cur_bd'].values, lw=0, color='lightsteelblue')
        
        axes2 = axes.twinx()
        axes2.set_ylabel('%gap')
        axes2.set_ylim(0, 100)
        line3, = axes2.plot(dfResult['time'].values, dfResult['gap'].values, color = 'red', label='Current Gap')
        axes.legend(handles=[line1, line2, line3], bbox_to_anchor=(0.5, 1.1), frameon=False, loc='upper center', ncol=3)
        
        plt.show()

    def plotLocation(df):
        fig, axes = plt.subplots()
        
        plt.scatter(df.loc[df['node'].str.contains('p'),'x'].values, df.loc[df['node'].str.contains('p'),'y'].values, s=50, facecolor='red', marker='o')
        plt.scatter(df.loc[df['node'].str.contains('d'),'x'].values, df.loc[df['node'].str.contains('d'),'y'].values, s=50, facecolor='green', marker='o')
        plt.scatter(df.loc[df['node'].str.contains('o'),'x'].values, df.loc[df['node'].str.contains('o'),'y'].values, s=50, facecolor='red', marker='s')
        plt.scatter(df.loc[df['node'].str.contains('e'),'x'].values, df.loc[df['node'].str.contains('e'),'y'].values, s=50, facecolor='green', marker='s')
        plt.scatter(df.loc[df['node'].str.contains('t'),'x'].values, df.loc[df['node'].str.contains('t'),'y'].values, s=50, facecolor='blue', marker='D')
        
        for xi, yi, text in zip(df['x'].values, df['y'].values, df['node'].values):
            plt.annotate(text, xy=(xi, yi), xycoords='data', xytext=(5, 5), textcoords='offset points')
        
        xResult = pd.DataFrame(x.keys(), columns=["k","i","j"])
        xResult["value"]=model.getAttr("X", x).values()
        for index, row in xResult.iterrows():
            if row["value"] == 1:
                x1 = df.loc[df['node'] == row["i"], 'x'].values
                y1 = df.loc[df['node'] == row["i"], 'y'].values
                x2 = df.loc[df['node'] == row["j"], 'x'].values
                y2 = df.loc[df['node'] == row["j"], 'y'].values
                plt.plot([x1, x2], [y1, y2], 'gray', linestyle="--")
        plt.show()
        
    plotGap(model._data)
    plotLocation(df)
    
    if model.Status == GRB.OPTIMAL:
        infos = [filename, model.getObjective().getValue(), model.Runtime]
    elif model.Status == GRB.TIME_LIMIT:
        if model.SolCount == 0:
            infos = [filename, "time_limit", model.Runtime]    
        else:
            infos = [filename, model.ObjVal, model.Runtime]    
    else:
        infos = [filename, "infisible", model.Runtime]  
        
    return infos

raisModel(filename)



