import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import time
import os
from tqdm import trange
from pyemd import emd
import argparse

def MSE(targetProb_, historicalProb_):
    return np.sqrt(np.square(targetProb_ - historicalProb_)).sum()

def loadDataOnly(data, BATCH_SIZE, queryOutputPath, reStore):
    trajectories = pd.read_csv(data, header = None)
    trajectories.columns = ['time', 'longitude', 'latitude', 'id']
    resid = int(((len(trajectories) / 60) // BATCH_SIZE) * BATCH_SIZE * 60)
    trajectories = trajectories[:resid]
    if reStore:
        with open(queryOutputPath, mode='w') as f:
            trajectories.to_csv(f, header = None, index = False)
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
                    temp =loadDataOnly(dataPath + file, BATCH_SIZE, None, False)
                    historicalTrajectories = pd.concat([historicalTrajectories, temp], axis=0)
    else:
        for i in range(int(day)-history, int(day)):
            for j in range(24):
                file = '{}_{}.csv'.format(i, j)
                if os.path.exists(dataPath + file):
                    temp = loadDataOnly(dataPath + file, BATCH_SIZE, None, False)
                    historicalTrajectories = pd.concat([historicalTrajectories, temp], axis=0)
        for i in range(int(day), int(day)+1):
            for j in range(int(hour)):
                file = '{}_{}.csv'.format(i, j)
                if os.path.exists(dataPath + file):
                    temp = loadDataOnly(dataPath + file, BATCH_SIZE, None, False)
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
    with open(retrievedTrajectories, mode = 'w') as f:
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


def main(args):
    BATCH_SIZE = 16
    history = 6
    path_ = '../results/{}/KDTree{}/'.format(args.METHOD, args.METHOD)
    if not os.path.exists(path_):
        os.mkdir(path_)
    path_ = '../results/{}/KDTree{}/EMD/'.format(args.METHOD, args.METHOD)
    if not os.path.exists(path_):
        os.mkdir(path_)
    historicalData = '../data/Experiment/history_data_before_time/'
    targetData = '../data/Experiment/query_data_before_time/8_17.csv'
    historicalScore_ = '../results/{}/'.format(args.METHOD)
    queryOutputPath = '../results/{}/KDTree{}/EMD/queryTrajectories.csv'.format(args.METHOD, args.METHOD)
    scoreFile = '../results/{}/KDTree{}/EMD/meanLoss.csv'.format(args.METHOD, args.METHOD)
    targetTrajectories = loadDataOnly(targetData, BATCH_SIZE, queryOutputPath, True)
    targetNum = 10
    historicalTrajectories = loadHistoricalDataOnly(historicalData, BATCH_SIZE, targetData, history)
    historicalScore = loadScore(historicalScore_, history)
        
    retrievedTrajectories = '../results/{}/KDTree{}/EMD/retrievedTrajectories.csv'.format(args.METHOD, args.METHOD)
    retrieval(scoreFile, historicalScore, targetNum, retrievedTrajectories, historicalTrajectories)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--METHOD", type=str, default="LCSS", choices=["LCSS","EDR","EDwP"], required=True)

    args = parser.parse_args()

    main(args)