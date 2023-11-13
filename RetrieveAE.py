import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import time
import os
from pyemd import emd
from tqdm import trange

embedding_dim = 16

def loadData(data, prob, BATCH_SIZE, queryOutputPath,reStore):
    trajectories = pd.read_csv(data, header = None)
    trajectories.columns = ['time', 'longitude', 'latitude', 'id']
    resid = int(((len(trajectories) / 60) // BATCH_SIZE) * BATCH_SIZE * 60)
    trajectories = trajectories[:resid]
    prob_ = pd.read_csv(prob, header = None)
    if int(len(trajectories)/60) != len(prob_):
        print('something wrong!')
    if reStore:
        with open(queryOutputPath, mode='w') as f:
            trajectories.to_csv(f, header = None, index = False)
    return trajectories, prob_

def loadHistoricalData(dataPath, probPath, BATCH_SIZE, targetData, history):
    day = targetData.split('/')[-1].split('_')[0]
    hour = targetData.split('/')[-1].split('_')[1].split('.')[0]
    historicalTrajectories = pd.DataFrame()
    historicalProb_ = pd.DataFrame()
    if int(hour) == 0:
        for i in range(int(day)-history, int(day)):
            for j in range(24):
                file = '{}_{}.csv'.format(i, j)
                probFile = 'prob_{}_{}.csv'.format(i, j)
                pathExist = os.path.exists(dataPath + file) and os.path.exists(probPath + probFile)
                if pathExist:
                    temp, tempProb_ =loadData(dataPath + file, probPath + probFile, BATCH_SIZE, None, False)
                    historicalTrajectories = pd.concat([historicalTrajectories, temp], axis=0)
                    historicalProb_ = pd.concat([historicalProb_, tempProb_], axis=0)
    else:
        for i in range(int(day)-history, int(day)):
            for j in range(24):
                file = '{}_{}.csv'.format(i, j)
                probFile = 'prob_{}_{}.csv'.format(i, j)
                pathExist = os.path.exists(dataPath + file) and os.path.exists(probPath + probFile)
                if pathExist:
                    temp, tempProb_ = loadData(dataPath + file, probPath + probFile, BATCH_SIZE, None, False)
                    historicalTrajectories = pd.concat([historicalTrajectories, temp], axis=0)
                    historicalProb_ = pd.concat([historicalProb_, tempProb_], axis=0)
        for i in range(int(day), int(day)+1):
            for j in range(int(hour)):
                file = '{}_{}.csv'.format(i, j)
                probFile = 'prob_{}_{}.csv'.format(i, j)
                pathExist = os.path.exists(dataPath + file) and os.path.exists(probPath + probFile)
                if pathExist:
                    temp, tempProb_ = loadData(dataPath + file, probPath + probFile, BATCH_SIZE, None, False)
                    historicalTrajectories = pd.concat([historicalTrajectories, temp], axis=0)
                    historicalProb_ = pd.concat([historicalProb_, tempProb_], axis=0)
    historicalTrajectories = historicalTrajectories.reset_index(drop=True)
    historicalProb_ = historicalProb_.reset_index(drop=True)
    return historicalTrajectories, historicalProb_

def selectTrajectories(retrievedTrajectories, historicalTrajectories, solution):
    with open(retrievedTrajectories, mode = 'w') as f:
        for i in range(len(solution)):
            historicalTrajectories[solution[i]*60:(solution[i]+1)*60].to_csv(f, header = None, index = False)
    return 0

def retrieval(scoreFile, targetProb_, historicalProb_, targetNum, retrievedTrajectories, historicalTrajectories):
    tree = KDTree(historicalProb_)
    solution = []
    wf = open(scoreFile, mode='w')
    for i in range(len(targetProb_)):
        nearest_dist, nearest_ind = tree.query(targetProb_[i].reshape((1,embedding_dim)), k=targetNum)
        meanLoss = nearest_dist[0].mean()
        wf.write(str(i) + ',' + str(meanLoss) + '\n')
        solution += list(nearest_ind[0])
    selectTrajectories(retrievedTrajectories, historicalTrajectories, solution)
    wf.close()
    return 0

if __name__ == '__main__':
    BATCH_SIZE = 16
    path_ = '../results/AE/KDTreeAE/'
    if not os.path.exists(path_):
        os.mkdir(path_)
    path_ = '../results/AE/KDTreeAE/EMD/'
    if not os.path.exists(path_):
        os.mkdir(path_)
    historicalData = '../data/Experiment/history_data_before_time/'
    historicalProb = '../results/AE/Index/History/prob/'
    targetData = '../data/Experiment/query_data_before_time/8_17.csv'
    targetProb = '../results/AE/Index/Query/prob/prob_8_17.csv'
    queryOutputPath = '../results/AE/KDTreeAE/EMD/queryTrajectories.csv'
    scoreFile = '../results/AE/KDTreeAE/EMD/meanLoss.csv'
    targetTrajectories, targetProb_ = loadData(targetData, targetProb, BATCH_SIZE, queryOutputPath, True)
    targetProb_ = targetProb_.values
    targetNum = 10
    historicalTrajectories, historicalProb_ = loadHistoricalData(historicalData, historicalProb, BATCH_SIZE, targetData, history=6)
    historicalProb_ = historicalProb_.values
        
    retrievedTrajectories = '../results/AE/KDTreeAE/EMD/retrievedTrajectories.csv'
    retrieval(scoreFile, targetProb_, historicalProb_, targetNum, retrievedTrajectories, historicalTrajectories)