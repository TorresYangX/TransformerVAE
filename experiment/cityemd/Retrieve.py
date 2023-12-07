import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import time
import os
from tqdm import trange
from pyemd import emd
import argparse

trajectory_length = 60

def MSE(targetProb_, historicalProb_):
    return np.sqrt(np.square(targetProb_ - historicalProb_)).sum()

def loadDataOnly(data, BATCH_SIZE):
    trajectories = pd.read_csv(data, header = None)
    trajectories.columns = ['time', 'longitude', 'latitude', 'id']
    resid = int(((len(trajectories) / trajectory_length) // BATCH_SIZE) * BATCH_SIZE * trajectory_length)
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
    container = np.load(path + 'history_6.npy')
    return container
        
def selectTrajectories(retrievedTrajectories, historicalTrajectories, solution):
    with open(retrievedTrajectories, mode = 'w') as f:
        for i in range(len(solution)):
            historicalTrajectories[solution[i]*trajectory_length:(solution[i]+1)*trajectory_length
                                   ].to_csv(f, header = None, index = False)
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
    historicalData = '../data/Experiment/experiment_data_before_time/'
    targetData = '../data/Experiment/experiment_data_before_time/{}_{}.csv'.format(args.day, args.hour)
    groundTruthPath = '../data/Experiment/groundTruth/groundTruth_{}.csv'.format(args.day)
    historicalScore_ = '../results/{}/'.format(args.METHOD)
    scoreFile = '../results/{}/KDTree{}/EMD/meanLoss.csv'.format(args.METHOD, args.METHOD)
    targetTrajectories = loadDataOnly(targetData, BATCH_SIZE)
    targerLen = len(targetTrajectories)
    groundTruthLen = len(pd.read_csv(groundTruthPath, header = None))
    print('target length: {}'.format(targerLen))
    print('ground truth length: {}'.format(groundTruthLen))
    targetNum = int(groundTruthLen / targerLen)
    print('target number: {}'.format(targetNum))
    historicalTrajectories = loadHistoricalDataOnly(historicalData, BATCH_SIZE, targetData, history)
    historicalScore = loadScore(historicalScore_, history)
        
    retrievedTrajectories = '../results/{}/KDTree{}/EMD/retrievedTrajectories.csv'.format(args.METHOD, args.METHOD)
    retrieval(scoreFile, historicalScore, targetNum, retrievedTrajectories, historicalTrajectories)
    retrievalLen = len(pd.read_csv(retrievedTrajectories, header = None))
    print('retrieval length: {}'.format(retrievalLen))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--METHOD", type=str, default="LCSS", choices=["LCSS","EDR","EDwP","DTW"], required=True)

    parser.add_argument('-d','--day', type=int, default=2, help='day', required=True)

    parser.add_argument('-hour','--hour', type=int, default=0, help='hour', required=True)

    args = parser.parse_args()

    main(args)