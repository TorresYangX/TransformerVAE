import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import time
import os
from pyemd import emd
from tqdm import trange
import argparse

embedding_dim = 16
trajectory_length = 30


def getStartIndex(historicalData, day, hour, BATCH_SIZE=16):
    startIndex = 0
    for i in range(day+1):
        for j in range(24):
            if i == day and j == hour:
                return startIndex//trajectory_length
            else:
                if os.path.exists(historicalData + '{}_{}.csv'.format(i, j)):
                    trajectories = pd.read_csv(historicalData+"{}_{}.csv".format(i,j), header = None)
                    resid = int(((len(trajectories) / trajectory_length) // BATCH_SIZE) * BATCH_SIZE * trajectory_length)
                    startIndex += resid
    return startIndex//trajectory_length




def loadData(data, BATCH_SIZE, day, hour,  **kwargs):
    trajectories = pd.read_csv(data+"{}_{}.csv".format(day,hour), header = None)
    trajectories.columns = ['time', 'longitude', 'latitude', 'id']
    resid = int(((len(trajectories) / trajectory_length) // BATCH_SIZE) * BATCH_SIZE * trajectory_length)
    trajectories = trajectories[:resid]
    output = {}
    for key in kwargs.keys():
        if key == 'Prob':
            prob_ = pd.read_csv(kwargs[key]+"prob_{}_{}.csv".format(day, hour), header = None)
            if int(len(trajectories)/trajectory_length) != len(prob_):
                print("Prob:", len(trajectories)/trajectory_length, len(prob_))
                print('something wrong!')
            output[key] = prob_
        elif key == 'Mu':
            mu_ = pd.read_csv(kwargs[key]+"mu_{}_{}.csv".format(day, hour), header = None)
            if int(len(trajectories)/trajectory_length) != len(mu_):
                print("Mu:", len(trajectories)/trajectory_length, len(mu_))
                print('something wrong!')
            output[key] = mu_
        elif key == 'Sigma':
            sigma_ = pd.read_csv(kwargs[key]+"sigma_{}_{}.csv".format(day, hour), header = None)
            if int(len(trajectories)/trajectory_length) != len(sigma_):
                print("Sigma:", len(trajectories)/trajectory_length, len(sigma_))
                print('something wrong!')
            output[key] = sigma_
        elif key == 'Pi':
            pi_ = pd.read_csv(kwargs[key]+"pi_{}_{}.csv".format(day, hour), header = None)
            if int(len(trajectories)/trajectory_length) != len(pi_):
                print("Pi:", len(trajectories)/trajectory_length, len(pi_))
                print('something wrong!')
            output[key] = pi_
        elif key == 'Alpha':
            alpha_ = pd.read_csv(kwargs[key]+"alpha_{}_{}.csv".format(day, hour), header = None)
            if int(len(trajectories)/trajectory_length) != len(alpha_):
                print("Alpha:", len(trajectories)/trajectory_length, len(alpha_))
                print('something wrong!')
            output[key] = alpha_
        else:
            print('wrong key!')
    return trajectories, output

def loadHistoricalData(dataPath, BATCH_SIZE, day, hour, history, **kwargs):
    historicalTrajectories = pd.DataFrame()
    historicalProb_ = pd.DataFrame()
    historicalMu_ = pd.DataFrame()
    historicalSigma_ = pd.DataFrame()
    historicalPi_ = pd.DataFrame()
    historicalAlpha_ = pd.DataFrame()
    for i in range(int(day)-history, int(day)):
        for j in range(24):
            file = '{}_{}.csv'.format(i, j)
            pathExist = os.path.exists(dataPath + file)
            for key in kwargs.keys():
                if key == 'Prob':
                    pathExist = pathExist and os.path.exists(kwargs[key] + 'prob_{}_{}.csv'.format(i, j))
                elif key == 'Mu':
                    pathExist = pathExist and os.path.exists(kwargs[key] + 'mu_{}_{}.csv'.format(i, j))
                elif key == 'Sigma':
                    pathExist = pathExist and os.path.exists(kwargs[key] + 'sigma_{}_{}.csv'.format(i, j))
                elif key == 'Pi':
                    pathExist = pathExist and os.path.exists(kwargs[key] + 'pi_{}_{}.csv'.format(i, j))
                elif key == 'Alpha':
                    pathExist = pathExist and os.path.exists(kwargs[key] + 'alpha_{}_{}.csv'.format(i, j))
                else:
                    print('wrong key!')
            if pathExist:
                temp, tempDict = loadData(dataPath, BATCH_SIZE, i, j, **kwargs)
                historicalTrajectories = pd.concat([historicalTrajectories, temp], axis=0)
                if 'Prob' in tempDict.keys():
                    historicalProb_ = pd.concat([historicalProb_, tempDict['Prob']], axis=0)
                if 'Mu' in tempDict.keys():
                    historicalMu_ = pd.concat([historicalMu_, tempDict['Mu']], axis=0)
                if 'Sigma' in tempDict.keys():
                    historicalSigma_ = pd.concat([historicalSigma_, tempDict['Sigma']], axis=0)
                if 'Pi' in tempDict.keys():
                    historicalPi_ = pd.concat([historicalPi_, tempDict['Pi']], axis=0)
                if 'Alpha' in tempDict.keys():
                    historicalAlpha_ = pd.concat([historicalAlpha_, tempDict['Alpha']], axis=0)
                
    for i in range(int(day), int(day)+1):
        for j in range(int(hour)+1):
            file = '{}_{}.csv'.format(i, j)
            pathExist = os.path.exists(dataPath + file)
            for key in kwargs.keys():
                if key == 'Prob':
                    pathExist = pathExist and os.path.exists(kwargs[key] + 'prob_{}_{}.csv'.format(i, j))
                elif key == 'Mu':
                    pathExist = pathExist and os.path.exists(kwargs[key] + 'mu_{}_{}.csv'.format(i, j))
                elif key == 'Sigma':
                    pathExist = pathExist and os.path.exists(kwargs[key] + 'sigma_{}_{}.csv'.format(i, j))
                elif key == 'Pi':
                    pathExist = pathExist and os.path.exists(kwargs[key] + 'pi_{}_{}.csv'.format(i, j))
                elif key == 'Alpha':
                    pathExist = pathExist and os.path.exists(kwargs[key] + 'alpha_{}_{}.csv'.format(i, j))
                else:
                    print('wrong key!')
            if pathExist:
                temp, tempDict = loadData(dataPath, BATCH_SIZE, i, j, **kwargs)
                historicalTrajectories = pd.concat([historicalTrajectories, temp], axis=0)
                if 'Prob' in tempDict.keys():
                    historicalProb_ = pd.concat([historicalProb_, tempDict['Prob']], axis=0)
                if 'Mu' in tempDict.keys():
                    historicalMu_ = pd.concat([historicalMu_, tempDict['Mu']], axis=0)
                if 'Sigma' in tempDict.keys():
                    historicalSigma_ = pd.concat([historicalSigma_, tempDict['Sigma']], axis=0)
                if 'Pi' in tempDict.keys():
                    historicalPi_ = pd.concat([historicalPi_, tempDict['Pi']], axis=0)
                if 'Alpha' in tempDict.keys():
                    historicalAlpha_ = pd.concat([historicalAlpha_, tempDict['Alpha']], axis=0)

    historicalTrajectories = historicalTrajectories.reset_index(drop=True)
    historicalProb_ = historicalProb_.reset_index(drop=True)
    historicalMu_ = historicalMu_.reset_index(drop=True)
    historicalSigma_ = historicalSigma_.reset_index(drop=True)
    historicalPi_ = historicalPi_.reset_index(drop=True)
    historicalAlpha_ = historicalAlpha_.reset_index(drop=True)
    return historicalTrajectories, historicalProb_, historicalMu_, historicalSigma_, historicalPi_, historicalAlpha_
        

def retrieval(scoreFile, targetProb_, historicalProb_, paraNum, startIndex):
    tree = KDTree(historicalProb_)
    averageRank = 0
    with open(scoreFile, 'w') as f:
        for i in trange(len(targetProb_)):
            nearest_dist, nearest_ind = tree.query(targetProb_[i].reshape((1,paraNum*embedding_dim)), k=len(historicalProb_))
            correspondingIndex = startIndex + i 
            rank = nearest_ind[0].tolist().index(correspondingIndex)
            f.write(str(rank)+'\n')
            averageRank += rank
        averageRank /= len(targetProb_)
        f.write('averageRank: {}'.format(averageRank))
    return 0


def main(args):
    BATCH_SIZE = 16
    path_ = '../SSM_KNN/{}/KDTree{}'.format(args.model, args.model)
    if not os.path.exists(path_):
        os.mkdir(path_)
    path_ = '../SSM_KNN/{}/KDTree{}/SSM_KNN/'.format(args.model, args.model)
    if not os.path.exists(path_):
        os.mkdir(path_)
    historicalData = '../data/Experiment/SSM_KNN/DataBase_2/data_before_time/'
    targetData = '../data/Experiment/SSM_KNN/DataBase_1/data_before_time/'
    para = {
        "target": {
            "AE": {
                'Prob':'../SSM_KNN/AE/Index/Database_1/prob/'
            },
            "VAE": {
                'Mu':'../SSM_KNN/VAE/Index/Database_1/mu/',
                'Sigma': '../SSM_KNN/VAE/Index/Database_1/sigma/'
            },
            "NVAE": {
                'Mu':'../SSM_KNN/NVAE/Index/Database_1/mu/',
                'Sigma': '../SSM_KNN/NVAE/Index/Database_1/sigma/',
                'Pi': '../SSM_KNN/NVAE/Index/Database_1/pi/',
                'Alpha': '../SSM_KNN/NVAE/Index/Database_1/alpha/'
            }
        },
        "historical": {
            "AE": {
                'Prob':'../SSM_KNN/AE/Index/Database_2/prob/'
            },
            "VAE": {
                'Mu':'../SSM_KNN/VAE/Index/Database_2/mu/',
                'Sigma': '../SSM_KNN/VAE/Index/Database_2/sigma/'
            },
            "NVAE": {
                'Mu':'../SSM_KNN/NVAE/Index/Database_2/mu/',
                'Sigma': '../SSM_KNN/NVAE/Index/Database_2/sigma/',
                'Pi': '../SSM_KNN/NVAE/Index/Database_2/pi/',
                'Alpha': '../SSM_KNN/NVAE/Index/Database_2/alpha/'
            }
        }
    }
    
    scoreFile = '../SSM_KNN/{}/KDTree{}/SSM_KNN/averageRank.csv'.format(args.model, args.model)
    targetTrajectories, targetDict = loadData(targetData, BATCH_SIZE, args.day, args.hour, **(para['target'][args.model]))
    targerLen = len(targetTrajectories)//trajectory_length
    print('target length: {}'.format(targerLen))
    targetProb_ = pd.concat(targetDict.values(), axis=1)
    targetProb_ = targetProb_.values
    historicalTrajectories, historicalProb_, historicalMu_, historicalSigma_, historicalPi_, historicalAlpha_ = loadHistoricalData(historicalData, BATCH_SIZE, 
                                                                                                                  args.day, args.hour, 
                                                                                                                  history=6, **(para['historical'][args.model]))
    print('historical length: {}'.format(len(historicalTrajectories)/trajectory_length))
    historicalProb_ = pd.concat([historicalProb_, historicalMu_, historicalSigma_, historicalPi_, historicalAlpha_], axis=1)
    historicalProb_ = historicalProb_.values
    startIndex = getStartIndex(historicalData, args.day, args.hour, BATCH_SIZE)
    print('start index: {}'.format(startIndex))
    retrieval(scoreFile, targetProb_, historicalProb_, len(para['target'][args.model]), startIndex)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model', type=str, default="AE", help='MODEL', choices=["AE", "VAE", "NVAE"], required=True)
    parser.add_argument('-d','--day', type=int, default=2, help='day', required=True)
    parser.add_argument('-hour','--hour', type=int, default=0, help='hour', required=True)
    args = parser.parse_args()
    main(args)
