import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import os
import time
from tqdm import trange
from pyemd import emd

embedding_dim = 16
trajectory_length = 61
latent_dim = embedding_dim * trajectory_length

def MSE(targetProb_, historicalProb_):
    return np.sqrt(np.square(targetProb_ - historicalProb_)).sum()

def loadData(data, mu, sigma, pi, alpha, BATCH_SIZE, queryOutputPath, reStore):
    trajectories = pd.read_csv(data, header = None)
    trajectories.columns = ['time', 'longitude', 'latitude', 'id']
    resid = int(((len(trajectories) / 60) // BATCH_SIZE) * BATCH_SIZE * 60)
    trajectories = trajectories[:resid]
    mu_ = pd.read_csv(mu, header = None)
    sigma_ = pd.read_csv(sigma, header = None)
    pi_ = pd.read_csv(pi, header = None)
    alpha_ = pd.read_csv(alpha, header = None)
    if (int(len(trajectories)/60) != len(mu_)) & (int(len(trajectories)/60) != len(sigma_)):
        print('something wrong!')
    if reStore:
        with open(queryOutputPath, mode='w') as f:
            trajectories.to_csv(f, header = None, index = False)
    return trajectories, mu_, sigma_, pi_, alpha_

def loadHistoricalData(dataPath, muPath, sigmaPath, piPath, alphaPath,  BATCH_SIZE, targetData, history):
    day = targetData.split('/')[-1].split('_')[0]
    hour = targetData.split('/')[-1].split('_')[1].split('.')[0]
    historicalTrajectories = pd.DataFrame()
    historicalMu_ = pd.DataFrame()
    historicalSigma_ = pd.DataFrame()
    historicalPi_ = pd.DataFrame()
    historicalAlpha_ = pd.DataFrame()
    if int(hour) == 0:
        for i in range(int(day)-history, int(day)):
            for j in range(24):
                file = '{}_{}.csv'.format(i, j)
                muFile = 'mu_{}_{}.csv'.format(i, j)
                sigmaFile = 'sigma_{}_{}.csv'.format(i, j)
                piFile = 'pi_{}_{}.csv'.format(i, j)
                alphaFile = 'alpha_{}_{}.csv'.format(i, j)
                pathExist = os.path.exists(dataPath + file) and os.path.exists(muPath + muFile) and os.path.exists(sigmaPath + sigmaFile) and os.path.exists(piPath + piFile) and os.path.exists(alphaPath + alphaFile)
                if pathExist:
                    temp, tempMu_, tempSigma_, tempPi_, tempAlpha_ =loadData(dataPath+file, muPath+muFile, sigmaPath+sigmaFile, piPath+piFile, alphaPath+alphaFile, BATCH_SIZE, None, False)
                    historicalTrajectories = pd.concat([historicalTrajectories, temp], axis=0)
                    historicalMu_ = pd.concat([historicalMu_, tempMu_], axis=0)
                    historicalSigma_ = pd.concat([historicalSigma_, tempSigma_], axis=0)
                    historicalPi_ = pd.concat([historicalPi_, tempPi_], axis=0)
                    historicalAlpha_ = pd.concat([historicalAlpha_, tempAlpha_], axis=0)
    else:
        for i in range(int(day)-history, int(day)):
            for j in range(24):
                file = '{}_{}.csv'.format(i, j)
                muFile = 'mu_{}_{}.csv'.format(i, j)
                sigmaFile = 'sigma_{}_{}.csv'.format(i, j)
                piFile = 'pi_{}_{}.csv'.format(i, j)
                alphaFile = 'alpha_{}_{}.csv'.format(i, j)
                pathExist = os.path.exists(dataPath + file) and os.path.exists(muPath + muFile) and os.path.exists(sigmaPath + sigmaFile) and os.path.exists(piPath + piFile) and os.path.exists(alphaPath + alphaFile)
                if pathExist:
                    temp, tempMu_, tempSigma_, tempPi_, tempAlpha_ =loadData(dataPath+file, muPath+muFile, sigmaPath+sigmaFile, 
                                                                             piPath+piFile, alphaPath+alphaFile, BATCH_SIZE, None, False)
                    historicalTrajectories = pd.concat([historicalTrajectories, temp], axis=0)
                    historicalMu_ = pd.concat([historicalMu_, tempMu_], axis=0)
                    historicalSigma_ = pd.concat([historicalSigma_, tempSigma_], axis=0)
                    historicalPi_ = pd.concat([historicalPi_, tempPi_], axis=0)
                    historicalAlpha_ = pd.concat([historicalAlpha_, tempAlpha_], axis=0)
        for i in range(int(day), int(day)+1):
            for j in range(int(hour)):
                file = '{}_{}.csv'.format(i, j)
                muFile = 'mu_{}_{}.csv'.format(i, j)
                sigmaFile = 'sigma_{}_{}.csv'.format(i, j)
                piFile = 'pi_{}_{}.csv'.format(i, j)
                alphaFile = 'alpha_{}_{}.csv'.format(i, j)
                pathExist = os.path.exists(dataPath + file) and os.path.exists(muPath + muFile) and os.path.exists(sigmaPath + sigmaFile) and os.path.exists(piPath + piFile) and os.path.exists(alphaPath + alphaFile)
                if pathExist:
                    temp, tempMu_, tempSigma_, tempPi_, tempAlpha_ =loadData(dataPath+file, muPath+muFile, sigmaPath+sigmaFile,
                                                                            piPath+piFile, alphaPath+alphaFile, BATCH_SIZE, None, False)
                    historicalTrajectories = pd.concat([historicalTrajectories, temp], axis=0)
                    historicalMu_ = pd.concat([historicalMu_, tempMu_], axis=0)
                    historicalSigma_ = pd.concat([historicalSigma_, tempSigma_], axis=0)
                    historicalPi_ = pd.concat([historicalPi_, tempPi_], axis=0)
                    historicalAlpha_ = pd.concat([historicalAlpha_, tempAlpha_], axis=0)
    historicalTrajectories = historicalTrajectories.reset_index(drop=True)
    historicalMu_ = historicalMu_.reset_index(drop=True)
    historicalSigma_ = historicalSigma_.reset_index(drop=True)
    historicalPi_ = historicalPi_.reset_index(drop=True)
    historicalAlpha_ = historicalAlpha_.reset_index(drop=True)
    return historicalTrajectories, historicalMu_, historicalSigma_, historicalPi_, historicalAlpha_
        
def selectTrajectories(retrievedTrajectories, historicalTrajectories, solution):
    with open(retrievedTrajectories, mode = 'w') as f:
        for i in trange(len(solution)):
            historicalTrajectories[solution[i]*60:(solution[i]+1)*60].to_csv(f, header = None, index = False)
    return 0

def retrieval(scoreFile, targetProb_, historicalProb_, targetNum, retrievedTrajectories, historicalTrajectories):
    tree = KDTree(historicalProb_)
    solution = []
    wf = open(scoreFile, mode='w')
    for i in trange(len(targetProb_)):
        nearest_dist, nearest_ind = tree.query(targetProb_[i].reshape((1,4*latent_dim)), k=targetNum)
        meanLoss = nearest_dist[0].mean()
        wf.write(str(i) + ',' + str(meanLoss) + '\n')
        solution += list(nearest_ind[0])
    selectTrajectories(retrievedTrajectories, historicalTrajectories, solution)
    wf.close()
    return 0

        
if __name__ == '__main__':
    BATCH_SIZE = 16
    path_ = '../results/VAE_nvib/KDTreeVAE_nvib'
    if not os.path.exists(path_):
        os.mkdir(path_)
    path_ = '../results/VAE_nvib/KDTreeVAE_nvib/EMD/'
    if not os.path.exists(path_):
        os.mkdir(path_)
    historicalData = '../data/Experiment/history_data_before_time/'
    historicalMu = '../results/VAE_nvib/Index/History/mu/'
    historicalSigma = '../results/VAE_nvib/Index/History/sigma/'
    historicalPi = '../results/VAE_nvib/Index/History/pi/'
    historicalAlpha = '../results/VAE_nvib/Index/History/alpha/'
    targetData = '../data/Experiment/query_data_before_time/8_17.csv'
    targetMu = '../results/VAE_nvib/Index/Query/mu/mu_8_17.csv'
    targetSigma = '../results/VAE_nvib/Index/Query/sigma/sigma_8_17.csv'
    targetPi = '../results/VAE_nvib/Index/Query/pi/pi_8_17.csv'
    targetAlpha = '../results/VAE_nvib/Index/Query/alpha/alpha_8_17.csv'
    queryOutputPath = '../results/VAE_nvib/KDTreeVAE_nvib/EMD/queryTrajectories.csv'
    scoreFile = '../results/VAE_nvib/KDTreeVAE_nvib/EMD/meanLoss.csv'
    targetTrajectories, targetMu_, targetSigma_, targetPi_, tragetAlpha_ = loadData(targetData, targetMu, targetSigma, 
                                                                                    targetPi, targetAlpha, 
                                                                                    BATCH_SIZE, queryOutputPath, True)
    targetProb_ = pd.concat([targetMu_, targetSigma_, targetPi_, tragetAlpha_], axis=1)
    targetProb_ = targetProb_.values
    targetNum = 10
    historicalTrajectories, historicalMu_, historicalSigma_, historicalPi_, historicalAlpha_ = loadHistoricalData(historicalData, historicalMu, historicalSigma, 
                                                                                                                  historicalPi, historicalAlpha,
                                                                                                                  BATCH_SIZE, targetData, history=6)
    historicalProb_ = pd.concat([historicalMu_, historicalSigma_, historicalPi_, historicalAlpha_], axis=1)
    historicalProb_ = historicalProb_.values
    print(historicalProb_.shape)
    retrievedTrajectories = '../results/VAE_nvib/KDTreeVAE_nvib/EMD/retrievedTrajectories.csv'
    retrieval(scoreFile, targetProb_, historicalProb_, targetNum, retrievedTrajectories, historicalTrajectories)