import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time

import gurobipy as gp
from gurobipy import GRB

filename = "./PDPTWT/4R-4K-4T-180L-0.txt"

# Read the meta-data of problem (number of requests, number of vehicles, number of transport stations, capability of vehicles)
def readMetaData(filename):
    metaData = pd.read_csv(filename, nrows=2, sep= '\t', on_bad_lines='skip')
    return metaData

# Read the instance's data (name of node, location (x, y), time-windows, load of the request)
def readDataframe(filename):
    df = pd.read_csv(filename, skiprows=3, sep='\t')
    # temp = [df['node'].str.contains("t") == True]
    for index, row in df.iterrows():
        if "t" in row['node']:
            copy = row
            copy['node'] = copy['node'].replace('t', 'ts')
            df = df._append(copy, ignore_index = True)
            copy['node'] = copy['node'].replace('ts', 'tf')
            df = df._append(copy, ignore_index = True)
    for index, row in df.iterrows():
        if "t" in row['node'] and "ts" not in row['node'] and "tf" not in row['node']:
            df = df.drop(index)
    df = df.reset_index(drop=True)
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
def calculateDistance(x1, x2, y1, y2):
    return math.sqrt((x2 - x1)**2 + (y2-y1)**2)

# one unit of distance can be traveled in one time unit
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
    # transferNodes = df.loc[df['node'].str.contains('t') and not df.loc['node'].str.contains("s") and not df.loc['node'].str.contains("f") , 'node']
    transferStart = df.loc[df['node'].str.contains('ts'),'node']
    transferFinish = df.loc[df['node'].str.contains('tf'), 'node']
    return {"a":allNodes, "ro":rOrigins, "rd":rDestinations, "vo":vOrigins, "vd":vDestinations, "ts":transferStart, "tf":transferFinish}

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

# Model
def cortesModel(filename):
    model = gp.Model()
    metaData = readMetaData(filename)
    df = readDataframe(filename)
    nodeList = getNodeList(df)

    arcs = []
    for i in nodeList['vo'].values:
        for j in nodeList['ro'].values:
            arcs.append((i,j))
        for j in nodeList['ts'].values:
            arcs.append((i,j))
        for j in nodeList['vd'].values:
            arcs.append((i, i.replace("o", "e")))

    for i in nodeList['ro'].values:
        for j in np.concatenate((nodeList['ro'].values, nodeList['rd'].values, nodeList['ts'].values)):
            if i != j:
                arcs.append((i,j))

    for i in nodeList['rd'].values:
        for j in np.concatenate((nodeList['ro'].values, nodeList['rd'].values, nodeList['vd'].values, nodeList['ts'].values)):
            if not (i == j or (j in nodeList['ro'].values and i == j.replace("p","d"))):
                arcs.append((i,j))

    for i in nodeList['ts'].values:
        for j in nodeList['tf'].values:
            if i == j.replace("f", "s"):
                arcs.append((i,j))

    for i in nodeList['tf'].values:
        for j in np.concatenate((nodeList['ro'].values, nodeList['rd'].values, nodeList['vd'].values, nodeList['ts'].values)):
            if  not (i == j.replace('s', 'f') and j in nodeList['ts'].values):
                arcs.append((i,j))

    arcs = list(dict.fromkeys(arcs))

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
    bigM = 999
    print(df)
    print(arcs)

    xIndex = [(k, i, j) for k in pd.RangeIndex(nVehicles) for (i,j) in arcs]
    zIndex = [(k, r, i) for k in pd.RangeIndex(nVehicles) for r in pd.RangeIndex(nRequests) for i in nodeList['a'].values]
    aIndex = [i for i in nodeList['a'].values]
    atsIndex = [(k, i) for k in pd.RangeIndex(nVehicles) for i in nodeList['ts'].values]
    atfIndex = [(k, i) for k in pd.RangeIndex(nVehicles) for i in nodeList['tf'].values]

    x = model.addVars(xIndex, vtype=GRB.BINARY, name='x')
    z = model.addVars(zIndex, vtype=GRB.BINARY, name='x')
    a = model.addVars(aIndex, vtype=GRB.INTEGER, name='a')
    ats = model.addVars(atsIndex, vtype=GRB.INTEGER, name='ats')
    atf = model.addVars(atfIndex, vtype=GRB.INTEGER, name='atf')
    
    model.setObjective(sum((c[i][j] * x[k, i, j]) for (i,j) in arcs for k in pd.RangeIndex(nVehicles)), GRB.MINIMIZE)
    model.update()

    ## Constaints                  
    for k in pd.RangeIndex(nVehicles):
        model.addConstr(sum(x[k,i,j] for i in nodeList['vo'].values for j in nodeList['a'].values if "o" + str(k) == i if (i,j) in arcs) == 1, name='constr1')
        model.addConstr(sum(x[k,i,j] for i in nodeList['a'].values for j in nodeList['vd'].values if "e" + str(k) == j if (i,j) in arcs) == 1, name= 'constr2')
        model.addConstr(sum(sum(x[k,i,j] for j in nodeList['a'].values if (i,j) in arcs) for i in nodeList['vo'].values ) <= 1, name= 'constr3')
        
        for i in np.concatenate((nodeList['ro'].values, nodeList['rd'].values)):
            model.addConstr(sum(x[k,i,j] for j in nodeList['a'].values if (i,j) in arcs) == sum(x[k,j,i] for j in nodeList['a'].values if (j,i) in arcs), name= 'constr4')
            
        for (i,j) in arcs:
            if i in nodeList["ts"].values and j in nodeList["tf"].values and i.replace("ts",'') == j.replace("tf",''):
                model.addConstr(sum(x[k,temp,i] for temp in nodeList['a'].values if (temp,i) in arcs) == x[k,i,j], name= 'constr5')
            
                model.addConstr(sum(x[k,j,temp] for temp in nodeList['a'].values if (j,temp) in arcs) == x[k,i,j], name= 'constr6')
        
    for i in nodeList['rd'].values:
        model.addConstr(sum(sum(x[k,j,i] for j in nodeList['a'].values if (j,i) in arcs) for k in pd.RangeIndex(nVehicles)) == 1, name= 'constr7')
        
    for i in nodeList['ro'].values:    
        model.addConstr(sum(sum(x[k,i,j] for j in nodeList['a'].values if (i,j) in arcs) for k in pd.RangeIndex(nVehicles)) == 1, name= 'constr8')

    for (i,j) in arcs:
        for k in pd.RangeIndex(nVehicles):
            for r in pd.RangeIndex(nRequests):
                if i == "o"+str(k) and j == "p"+str(r):
                    model.addConstr((x[k,i,j] == 1) >> (a[j] >= c[i][j]), name="constr9")
            if i == "o"+str(k) and j in nodeList["ts"].values:
                model.addConstr((x[k,i,j] == 1) >> (ats[k,j] >= c[i][j]), name="constr10")

    for (i,j) in arcs:
        if i in np.concatenate((nodeList['ro'].values, nodeList['rd'].values)):  
            if j in np.concatenate((nodeList['ro'].values, nodeList['rd'].values)):
                for k in pd.RangeIndex(nVehicles):
                    model.addConstr((x[k, i, j] == 1) >> (a[i] + c[i][j] <= a[j]), name="constr11")
                    
            if j in nodeList["ts"].values:
                for k in pd.RangeIndex(nVehicles):
                    model.addConstr((x[k,i,j] == 1) >> (a[i] + c[i][j] <= ats[k,j]), name="constr12")
                    
        if j in np.concatenate((nodeList['ro'].values, nodeList['rd'].values)):              
            if i in nodeList["tf"].values:
                for k in pd.RangeIndex(nVehicles):
                    model.addConstr((x[k,i,j] == 1) >> (atf[k,i] + c[i][j] <= a[j]), name="constr14")
        
        if i in nodeList["ts"].values and j in nodeList["tf"].values:
            if i.replace('ts','') == j.replace('tf',''):
                for k in pd.RangeIndex(nVehicles):
                    model.addConstr((x[k,i,j] == 1) >> (ats[k,i] + c[i][j] <= atf[k,j]), name="constr13")
                
        if i in nodeList["tf"].values and j in nodeList["ts"].values:
            if i.replace("tf",'') != j.replace("ts",''):
                for k in pd.RangeIndex(nVehicles):
                    model.addConstr((x[k,i,j] == 1) >> (atf[k,i] + c[i][j] <= ats[k,j]), name="constr15")  
                              
    for k in pd.RangeIndex(nVehicles):
        for r in pd.RangeIndex(nRequests):
            model.addConstr(z[k,r,"o"+str(k)] == 0, name ="constr16")
            model.addConstr(z[k,r,"e"+str(k)] == 0, name ="constr16")
                
    for (i,j) in arcs:
        if not (i in nodeList["ts"].values and j in nodeList["tf"].values and i.replace("ts",'') == j.replace("tf",'')):
            for k in pd.RangeIndex(nVehicles):
                for r in pd.RangeIndex(nRequests):
                    if "p" + str(r) != i and "d" + str(r) != i:
                        model.addConstr((x[k,i,j] == 1) >> (z[k,r,i] <= z[k,r,j]), name="constr17")
                    
    for k in pd.RangeIndex(nVehicles):
        for r in pd.RangeIndex(nRequests):
            for (i,j) in arcs:
                if i == "p" + str(r):
                    model.addConstr((x[k,i,j] == 1) >> (z[k,r,j] == 1), name="constr18")
                if i == "d" + str(r):
                    model.addConstr((x[k,i,j] == 1) >> (z[k,r,j] == 0), name="constr19")
                    
    for r in pd.RangeIndex(nRequests):
        for (i,j) in arcs:
            if i in nodeList["ts"].values and j in nodeList["tf"].values and i.replace("ts",'') == j.replace("tf",''):
                model.addConstr(sum(z[k,r,i] for k in pd.RangeIndex(nVehicles)) == sum(z[k,r,j] for k in pd.RangeIndex(nVehicles)), name="constr20")
        

    for k in pd.RangeIndex(nVehicles):
        for j in nodeList["a"].values:
            if "o" + str(k) != j and "e" + str(k) != j:
                model.addConstr(sum(z[k,r,j] for r in pd.RangeIndex(nRequests)) <= nRequests * sum(x[k,i,j] for i in nodeList["a"].values if (i,j) in arcs), name="constr21") 

    for i in nodeList["ts"].values:
        for j in nodeList["tf"].values:
            if i.replace("ts",'') == j.replace("tf",''):
                for k1 in pd.RangeIndex(nVehicles):
                    for k2 in pd.RangeIndex(nVehicles):
                        for r in pd.RangeIndex(nRequests):
                            model.addConstr(ats[k1,i] - atf[k2,j] <= bigM * (2 - (z[k1,r,i] + z[k2,r,j])), name="constr22")

    for k in pd.RangeIndex(nVehicles):
        for i in nodeList["a"].values:
            model.addConstr(sum(q[r]*z[k,r,i] for r in pd.RangeIndex(nRequests)) <= u[k], name="constr23")
            
    for i in np.concatenate((nodeList['ro'].values, nodeList['rd'].values)):  
        model.addConstr(a[i] >= int(df.loc[df['node'] == i, 'a'].values[0]), name='constr24')
        model.addConstr(a[i] <= int(df.loc[df['node'] == i, 'b'].values[0]), name='constr24')											
        
                        
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
        fig, axes = plt.subplots(figsize=(10, 10))
        
        plt.scatter(df.loc[df['node'].str.contains('p'),'x'].values, df.loc[df['node'].str.contains('p'),'y'].values, s=50, facecolor='red', marker='o')
        plt.scatter(df.loc[df['node'].str.contains('d'),'x'].values, df.loc[df['node'].str.contains('d'),'y'].values, s=50, facecolor='green', marker='o')
        plt.scatter(df.loc[df['node'].str.contains('o'),'x'].values, df.loc[df['node'].str.contains('o'),'y'].values, s=50, facecolor='red', marker='s')
        plt.scatter(df.loc[df['node'].str.contains('e'),'x'].values, df.loc[df['node'].str.contains('e'),'y'].values, s=50, facecolor='green', marker='s')
        plt.scatter(df.loc[df['node'].str.contains('ts'),'x'].values, df.loc[df['node'].str.contains('ts'),'y'].values, s=50, facecolor='blue', marker='D')
        # plt.scatter(df.loc[df['node'].str.contains('tf'),'x'].values, df.loc[df['node'].str.contains('tf'),'y'].values, s=50, facecolor='blue', marker='D')
        
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
    
    def plotArcs(arcs):
        fig, axes = plt.subplots()
        
        plt.scatter(df.loc[df['node'].str.contains('p'),'x'].values, df.loc[df['node'].str.contains('p'),'y'].values, s=50, facecolor='red', marker='o')
        plt.scatter(df.loc[df['node'].str.contains('d'),'x'].values, df.loc[df['node'].str.contains('d'),'y'].values, s=50, facecolor='green', marker='o')
        plt.scatter(df.loc[df['node'].str.contains('o'),'x'].values, df.loc[df['node'].str.contains('o'),'y'].values, s=50, facecolor='red', marker='s')
        plt.scatter(df.loc[df['node'].str.contains('e'),'x'].values, df.loc[df['node'].str.contains('e'),'y'].values, s=50, facecolor='green', marker='s')
        plt.scatter(df.loc[df['node'].str.contains('ts'),'x'].values, df.loc[df['node'].str.contains('ts'),'y'].values, s=50, facecolor='blue', marker='D')
        
        for (i,j) in arcs:
            x1 = df.loc[df['node'] == i, 'x'].values
            y1 = df.loc[df['node'] == i, 'y'].values
            x2 = df.loc[df['node'] == j, 'x'].values
            y2 = df.loc[df['node'] == j, 'y'].values
            plt.plot([x1, x2], [y1, y2], 'gray', linestyle="--")
            
        plt.show()
        
    plotGap(model._data)
    plotLocation(df)
    # plotArcs(arcs)
    
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


cortesModel(filename)