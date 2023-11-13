import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import os
import time
from tqdm import trange
from pyemd import emd

embedding_dim = 16


def loadData(data, mu, sigma, BATCH_SIZE, queryOutputPath,reStore):
    trajectories = pd.read_csv(data, header = None)
    trajectories.columns = ['time', 'longitude', 'latitude', 'id']
    resid = int(((len(trajectories) / 60) // BATCH_SIZE) * BATCH_SIZE * 60)
    trajectories = trajectories[:resid]
    mu_ = pd.read_csv(mu, header = None)
    sigma_ = pd.read_csv(sigma, header = None)
    if (int(len(trajectories)/60) != len(mu_)) & (int(len(trajectories)/60) != len(sigma_)):
        print('something wrong!')
    if reStore:
        with open(queryOutputPath, mode='w') as f:
            trajectories.to_csv(f, header = None, index = False)
    return trajectories, mu_, sigma_

def loadHistoricalData(dataPath, muPath, sigmaPath, BATCH_SIZE, targetData, history):
    day = targetData.split('/')[-1].split('_')[0]
    hour = targetData.split('/')[-1].split('_')[1].split('.')[0]
    historicalTrajectories = pd.DataFrame()
    historicalMu_ = pd.DataFrame()
    historicalSigma_ = pd.DataFrame()
    if int(hour) == 0:
        for i in range(int(day)-history, int(day)):
            for j in range(24):
                file = '{}_{}.csv'.format(i, j)
                muFile = 'mu_{}_{}.csv'.format(i, j)
                sigmaFile = 'sigma_{}_{}.csv'.format(i, j)
                pathExist = os.path.exists(dataPath + file) and os.path.exists(muPath + muFile) and os.path.exists(sigmaPath + sigmaFile)
                if pathExist:
                    temp, tempMu_, tempSigma_ =loadData(dataPath + file, muPath + muFile, sigmaPath + sigmaFile, BATCH_SIZE, None, False)
                    historicalTrajectories = pd.concat([historicalTrajectories, temp], axis=0)
                    historicalMu_ = pd.concat([historicalMu_, tempMu_], axis=0)
                    historicalSigma_ = pd.concat([historicalSigma_, tempSigma_], axis=0)
    else:
        for i in range(int(day)-history, int(day)):
            for j in range(24):
                file = '{}_{}.csv'.format(i, j)
                muFile = 'mu_{}_{}.csv'.format(i, j)
                sigmaFile = 'sigma_{}_{}.csv'.format(i, j)
                pathExist = os.path.exists(dataPath + file) and os.path.exists(muPath + muFile) and os.path.exists(sigmaPath + sigmaFile)
                if pathExist:
                    temp, tempMu_, tempSigma_ =loadData(dataPath + file, muPath + muFile, sigmaPath + sigmaFile, BATCH_SIZE, None, False)
                    historicalTrajectories = pd.concat([historicalTrajectories, temp], axis=0)
                    historicalMu_ = pd.concat([historicalMu_, tempMu_], axis=0)
                    historicalSigma_ = pd.concat([historicalSigma_, tempSigma_], axis=0)
        for i in range(int(day), int(day)+1):
            for j in range(int(hour)):
                file = '{}_{}.csv'.format(i, j)
                muFile = 'mu_{}_{}.csv'.format(i, j)
                sigmaFile = 'sigma_{}_{}.csv'.format(i, j)
                pathExist = os.path.exists(dataPath + file) and os.path.exists(muPath + muFile) and os.path.exists(sigmaPath + sigmaFile)
                if pathExist:
                    temp, tempMu_, tempSigma_ =loadData(dataPath + file, muPath + muFile, sigmaPath + sigmaFile, BATCH_SIZE, None, False)
                    historicalTrajectories = pd.concat([historicalTrajectories, temp], axis=0)
                    historicalMu_ = pd.concat([historicalMu_, tempMu_], axis=0)
                    historicalSigma_ = pd.concat([historicalSigma_, tempSigma_], axis=0)
    historicalTrajectories = historicalTrajectories.reset_index(drop=True)
    historicalMu_ = historicalMu_.reset_index(drop=True)
    historicalSigma_ = historicalSigma_.reset_index(drop=True)
    return historicalTrajectories, historicalMu_, historicalSigma_
        
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
        nearest_dist, nearest_ind = tree.query(targetProb_[i].reshape((1, embedding_dim*2)), k=targetNum)
        meanLoss = nearest_dist[0].mean()
        wf.write(str(i) + ',' + str(meanLoss) + '\n')
        solution += list(nearest_ind[0])
    selectTrajectories(retrievedTrajectories, historicalTrajectories, solution)
    wf.close()
    return 0


def normalize(targetTrajectories):
    targetX = targetTrajectories[['latitude', 'longitude']]
    targetX.loc[:, 'latitude'] = (targetX.loc[:, 'latitude'] - 39.9) / 0.3
    targetX.loc[:, 'longitude'] = (targetX.loc[:, 'longitude'] - 116.4) / 0.4
    targetX = targetX.values.reshape(-1, 60, 2)
    return targetX

def cityEMD(targetTrajectories, retrievedTrajectories_,  NLAT=6, NLON=8):
    targetX = normalize(targetTrajectories)
    retrievedY = normalize(retrievedTrajectories_)
    Xdis = ((0.5 * (targetX[:, :, 0] + 1) * NLAT).astype(int) * NLON + (0.5 * (targetX[:, :, 1] + 1) * NLON).astype(int)).astype(int)
    Ydis = ((0.5 * (retrievedY[:, :, 0] + 1) * NLAT).astype(int) * NLON + (0.5 * (retrievedY[:, :, 1] + 1) * NLON).astype(int)).astype(int)
    flowReal = np.zeros((NLAT*NLON,NLAT*NLON,59))
    flowRetrieved = np.zeros((NLAT*NLON,NLAT*NLON,59))
    flowDistance = np.zeros((NLAT*NLON,NLAT*NLON))
    for i in trange(NLAT*NLON):
        for j in range(NLAT*NLON):
            flowDistance[i, j] = min(10.0, np.sqrt(((i//NLON)-(j//NLON))**2+((i%NLON)-(j%NLON))**2))
            for k in range(59):
                flowReal[Xdis[i,k],Xdis[j,k+1],k] += 1.0
                flowRetrieved[Ydis[i,k],Ydis[j,k+1],k] += 1.0
    emd_ = np.zeros(59)
    for kt in range(59):
        for it in range(NLAT*NLON):
            emd_[kt] += emd(flowReal[it, :, kt].copy(order='C'), flowRetrieved[it, :, kt].copy(order='C'), flowDistance)
    np.save('../results/VAE/KDTreeVAE/EMD/emd_.npy', emd_)
    return 0
        
if __name__ == '__main__':
    BATCH_SIZE = 16
    path_ = '../results/VAE/KDTreeVAE/'
    if not os.path.exists(path_):
        os.mkdir(path_)
    path_ = '../results/VAE/KDTreeVAE/EMD/'
    if not os.path.exists(path_):
        os.mkdir(path_)
    historicalData = '../data/Experiment/history_data_before_time/'
    historicalMu = '../results/VAE/Index/History/mu/'
    historicalSigma = '../results/VAE/Index/History/sigma/'
    targetData = '../data/Experiment/query_data_before_time/8_17.csv'
    targetMu = '../results/VAE/Index/Query/mu/mu_8_17.csv'
    targetSigma = '../results/VAE/Index/Query/sigma/sigma_8_17.csv'
    queryOutputPath = '../results/VAE/KDTreeVAE/EMD/queryTrajectories.csv'
    scoreFile = '../results/VAE/KDTreeVAE/EMD/meanLoss.csv'
    targetTrajectories, targetMu_, targetSigma_ = loadData(targetData, targetMu, targetSigma, BATCH_SIZE, queryOutputPath, True)
    targetProb_ = pd.concat([targetMu_, targetSigma_], axis=1)
    targetProb_ = targetProb_.values
    targetNum = 10
    historicalTrajectories, historicalMu_, historicalSigma_ = loadHistoricalData(historicalData, historicalMu, historicalSigma, BATCH_SIZE, targetData, history=6)
    historicalProb_ = pd.concat([historicalMu_, historicalSigma_], axis=1)
    historicalProb_ = historicalProb_.values
    retrievedTrajectories = '../results/VAE/KDTreeVAE/EMD/retrievedTrajectories.csv'
    retrieval(scoreFile, targetProb_, historicalProb_, targetNum, retrievedTrajectories, historicalTrajectories)
    
    retrievedTrajectories_ = pd.read_csv(retrievedTrajectories, header = None)
    retrievedTrajectories_.columns = ['time', 'longitude', 'latitude', 'id']
    cityEMD(targetTrajectories, retrievedTrajectories_)