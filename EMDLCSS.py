import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import time
import os
from traj_simi import LCSS
import multiprocessing as mp
from pyemd import emd

def MSE(targetProb_, historicalProb_):
    return np.sqrt(np.square(targetProb_ - historicalProb_)).sum()

def loadDataOnly(data, BATCH_SIZE):
    trajectories = pd.read_csv(data, header = None)
    trajectories.columns = ['time', 'longitude', 'latitude', 'id']
    resid = int(((len(trajectories) / 60) // BATCH_SIZE) * BATCH_SIZE * 60)
    trajectories = trajectories[:resid]
    return trajectories

def loadHistoricalDataOnly(dataPath, BATCH_SIZE, targetData, history):
    day = targetData.split('/')[-1].split('_')[0]
    hour = targetData.split('/')[-1].split('_')[1].split('.')[0]
    historicalTrajectories = pd.DataFrame()
    if int(hour) == 0:
        for i in range(int(day)-history, int(day)):
            for j in range(24):
                file = '{}_{}.csv'.format(i, j)
                if os.path.exists(dataPath + file):
                    temp =loadDataOnly(dataPath + file, BATCH_SIZE)
                    historicalTrajectories = pd.concat([historicalTrajectories, temp], axis=0)
    else:
        for i in range(int(day)-history, int(day)):
            for j in range(24):
                file = '{}_{}.csv'.format(i, j)
                if os.path.exists(dataPath + file):
                    temp = loadDataOnly(dataPath + file, BATCH_SIZE)
                    historicalTrajectories = pd.concat([historicalTrajectories, temp], axis=0)
        for i in range(int(day), int(day)+1):
            for j in range(int(hour)):
                file = '{}_{}.csv'.format(i, j)
                if os.path.exists(dataPath + file):
                    temp = loadDataOnly(dataPath + file, BATCH_SIZE)
                    historicalTrajectories = pd.concat([historicalTrajectories, temp], axis=0)
    historicalTrajectories = historicalTrajectories.reset_index(drop=True)
    return historicalTrajectories

def loadScore(path, history):
    container = np.load(path + 'history_1.npy')
    for i in range(2, int(history+1)):
        dataPath = path + 'history_{}.npy'.format(i)
        temp = np.load(dataPath)
        container = np.append(container, temp, axis=1)
    return container
        
def selectTrajectories(retrievedTrajectories, historicalTrajectories, solution):
    with open(retrievedTrajectories, mode = 'a') as f:
        for i in range(len(solution)):
            historicalTrajectories[solution[i]*60:(solution[i]+1)*60].to_csv(f, header = None, index = False)
    return 0

def retrieval(scoreFile, historicalScore, targetNum, retrievedTrajectories, historicalTrajectories):
    solution = []
    wf = open(scoreFile, mode='w')
    for i in range(len(historicalScore)):
        nearest_dist = historicalScore[i, np.argpartition(historicalScore[i], range(targetNum))[:targetNum]]
        nearest_ind = np.argpartition(historicalScore[i], range(targetNum))[:targetNum]
        meanLoss = nearest_dist[0].mean()
        wf.write(str(i) + ',' + str(meanLoss) + '\n')
        solution += list(nearest_ind)
    selectTrajectories(retrievedTrajectories, historicalTrajectories, solution)
    wf.close()
    return 0

def normalize(targetTrajectories):
    targetX = targetTrajectories[['latitude', 'longitude']]
    targetX.loc[:, 'latitude'] = (targetX.loc[:, 'latitude'] - 39.9) / 0.3
    targetX.loc[:, 'longitude'] = (targetX.loc[:, 'longitude'] - 116.4) / 0.4
    targetX = targetX.values.reshape(-1, 60, 2)
    return targetX

def cityEMD(targetTrajectories, reterievedTrajectories_, thresholdDistance, latS=40.2, latN=39.6, lonW=116.0, lonE=116.8, NLAT=8, NLON=8):
    targetX = normalize(targetTrajectories)
    retrievedY = normalize(reterievedTrajectories_)
    Xdis = ((0.5 * (targetX[:, :, 0] + 1) * NLAT).astype(int) * NLON + (0.5 * (targetX[:, :, 1] + 1) * NLON).astype(int)).astype(int)
    Ydis = ((0.5 * (retrievedY[:, :, 0] + 1) * NLAT).astype(int) * NLON + (0.5 * (retrievedY[:, :, 1] + 1) * NLON).astype(int)).astype(int)
    flowReal = np.zeros((NLAT*NLON,NLAT*NLON,59))
    flowRetrieved = np.zeros((NLAT*NLON,NLAT*NLON,59))
    flowDistance = np.zeros((NLAT*NLON,NLAT*NLON))
    for i in range(64):
        print(i, ' / 64...', time.ctime())
        for j in range(64):
            flowDistance[i, j] = min(10.0, np.sqrt(((i//NLON)-(j//NLON))**2+((i%NLON)-(j%NLON))**2))
            for k in range(59):
                flowReal[Xdis[i,k],Xdis[j,k+1],k] += 1.0
                flowRetrieved[Ydis[i,k],Ydis[j,k+1],k] += 1.0
    emd_ = np.zeros(59)
    for kt in range(59):
        print(kt, ' / 59...', time.ctime())
        for it in range(64):
            emd_[kt] += emd(flowReal[it, :, kt].copy(order='C'), flowRetrieved[it, :, kt].copy(order='C'), flowDistance)
    np.save('../small_results/LCSS/KDTreeLCSS/EMD/emd_.npy', emd_)
    return 0

if __name__ == '__main__':
    BATCH_SIZE = 16
    history = 6
    path_ = '../small_results/LCSS/KDTreeLCSS/EMD/'
    if not os.path.exists(path_):
        os.mkdir(path_)
    historicalData = '../small_data/data_before_time/'
    targetData = '../small_data/query_data_before_time/8_7.csv'
    historicalScore_ = '../small_results/LCSS/'
    scoreFile = '../small_results/LCSS/KDTreeLCSS/EMD/meanLoss.csv'.format(history)
    targetTrajectories = loadDataOnly(targetData, BATCH_SIZE)
    targetNum = 1
    historicalTrajectories = loadHistoricalDataOnly(historicalData, BATCH_SIZE, targetData, history)
    historicalScore = loadScore(historicalScore_, history)
    print('finish loading data', time.ctime())
        
    retrievedTrajectories = '../small_results/LCSS/KDTreeLCSS/EMD/retrievedTrajectories.csv'.format(history)
    retrieval(scoreFile, historicalScore, targetNum, retrievedTrajectories, historicalTrajectories)
    print('finish retrieval', time.ctime())
    
    retrievedTrajectories_ = pd.read_csv(retrievedTrajectories, header = None)
    retrievedTrajectories_.columns = ['time', 'longitude', 'latitude', 'id']
    cityEMD(targetTrajectories, retrievedTrajectories_, thresholdDistance=10)