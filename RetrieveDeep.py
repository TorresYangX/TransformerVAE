import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import time
import os
from pyemd import emd
from tqdm import trange
import argparse

embedding_dim = 16
trajectory_length = 60


def loadData(data, BATCH_SIZE, **kwargs):
    trajectories = pd.read_csv(data, header = None)
    trajectories.columns = ['time', 'longitude', 'latitude', 'id']
    resid = int(((len(trajectories) / 60) // BATCH_SIZE) * BATCH_SIZE * 60)
    trajectories = trajectories[:resid]
    ## return a dict of output according to the kwargs.keys()
    output = {}
    for key in kwargs.keys():
        if key == 'targetProb':
            prob_ = pd.read_csv(kwargs[key], header = None)
            if int(len(trajectories)/trajectory_length) != len(prob_):
                print('something wrong!')
            output[key] = prob_
        elif key == 'targetMu':
            mu_ = pd.read_csv(kwargs[key], header = None)
            if int(len(trajectories)/trajectory_length) != len(mu_):
                print('something wrong!')
            output[key] = mu_
        elif key == 'targetSigma':
            sigma_ = pd.read_csv(kwargs[key], header = None)
            if int(len(trajectories)/trajectory_length) != len(sigma_):
                print('something wrong!')
            output[key] = sigma_
        elif key == 'targetPi':
            pi_ = pd.read_csv(kwargs[key], header = None)
            if int(len(trajectories)/trajectory_length) != len(pi_):
                print('something wrong!')
            output[key] = pi_
        elif key == 'targetAlpha':
            alpha_ = pd.read_csv(kwargs[key], header = None)
            if int(len(trajectories)/trajectory_length) != len(alpha_):
                print('something wrong!')
            output[key] = alpha_
        else:
            print('wrong key!')
    return trajectories, output

def loadHistoricalData(dataPath, BATCH_SIZE, targetData, history, **kwargs):
    day = targetData.split('/')[-1].split('_')[0]
    hour = targetData.split('/')[-1].split('_')[1].split('.')[0]
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
                if key == 'historicalProb':
                    pathExist = pathExist and os.path.exists(kwargs[key] + 'prob_{}_{}.csv'.format(i, j))
                elif key == 'historicalMu':
                    pathExist = pathExist and os.path.exists(kwargs[key] + 'mu_{}_{}.csv'.format(i, j))
                elif key == 'historicalSigma':
                    pathExist = pathExist and os.path.exists(kwargs[key] + 'sigma_{}_{}.csv'.format(i, j))
                elif key == 'historicalPi':
                    pathExist = pathExist and os.path.exists(kwargs[key] + 'pi_{}_{}.csv'.format(i, j))
                elif key == 'historicalAlpha':
                    pathExist = pathExist and os.path.exists(kwargs[key] + 'alpha_{}_{}.csv'.format(i, j))
                else:
                    print('wrong key!')
            if pathExist:
                for key in kwargs.keys():
                    if key == 'historicalProb':
                        temp_ = pd.read_csv(kwargs[key] + 'prob_{}_{}.csv'.format(i, j), header = None)
                        historicalProb_ = pd.concat([historicalProb_, temp_], axis=0)
                    elif key == 'historicalMu':
                        tempMu_ = pd.read_csv(kwargs[key] + 'mu_{}_{}.csv'.format(i, j), header = None)
                        historicalMu_ = pd.concat([historicalMu_, tempMu_], axis=0)
                    elif key == 'historicalSigma':
                        tempSigma_ = pd.read_csv(kwargs[key] + 'sigma_{}_{}.csv'.format(i, j), header = None)
                        historicalSigma_ = pd.concat([historicalSigma_, tempSigma_], axis=0)
                    elif key == 'historicalPi':
                        tempPi_ = pd.read_csv(kwargs[key] + 'pi_{}_{}.csv'.format(i, j), header = None)
                        historicalPi_ = pd.concat([historicalPi_, tempPi_], axis=0)
                    elif key == 'historicalAlpha':
                        tempAlpha_ = pd.read_csv(kwargs[key] + 'alpha_{}_{}.csv'.format(i, j), header = None)
                        historicalAlpha_ = pd.concat([historicalAlpha_, tempAlpha_], axis=0)
                    else:
                        print('wrong key!')
    for i in range(int(day), int(day)+1):
        for j in range(int(hour)):
            file = '{}_{}.csv'.format(i, j)
            pathExist = os.path.exists(dataPath + file)
            for key in kwargs.keys():
                if key == 'historicalProb':
                    pathExist = pathExist and os.path.exists(kwargs[key] + 'prob_{}_{}.csv'.format(i, j))
                elif key == 'historicalMu':
                    pathExist = pathExist and os.path.exists(kwargs[key] + 'mu_{}_{}.csv'.format(i, j))
                elif key == 'historicalSigma':
                    pathExist = pathExist and os.path.exists(kwargs[key] + 'sigma_{}_{}.csv'.format(i, j))
                elif key == 'historicalPi':
                    pathExist = pathExist and os.path.exists(kwargs[key] + 'pi_{}_{}.csv'.format(i, j))
                elif key == 'historicalAlpha':
                    pathExist = pathExist and os.path.exists(kwargs[key] + 'alpha_{}_{}.csv'.format(i, j))
                else:
                    print('wrong key!')
            if pathExist:
                for key in kwargs.keys():
                    if key == 'historicalProb':
                        temp_ = pd.read_csv(kwargs[key] + 'prob_{}_{}.csv'.format(i, j), header = None)
                        historicalProb_ = pd.concat([historicalProb_, temp_], axis=0)
                    elif key == 'historicalMu':
                        tempMu_ = pd.read_csv(kwargs[key] + 'mu_{}_{}.csv'.format(i, j), header = None)
                        historicalMu_ = pd.concat([historicalMu_, tempMu_], axis=0)
                    elif key == 'historicalSigma':
                        tempSigma_ = pd.read_csv(kwargs[key] + 'sigma_{}_{}.csv'.format(i, j), header = None)
                        historicalSigma_ = pd.concat([historicalSigma_, tempSigma_], axis=0)
                    elif key == 'historicalPi':
                        tempPi_ = pd.read_csv(kwargs[key] + 'pi_{}_{}.csv'.format(i, j), header = None)
                        historicalPi_ = pd.concat([historicalPi_, tempPi_], axis=0)
                    elif key == 'historicalAlpha':
                        tempAlpha_ = pd.read_csv(kwargs[key] + 'alpha_{}_{}.csv'.format(i, j), header = None)
                        historicalAlpha_ = pd.concat([historicalAlpha_, tempAlpha_], axis=0)
                    else:
                        print('wrong key!')
    historicalTrajectories = historicalTrajectories.reset_index(drop=True)
    historicalProb_ = historicalProb_.reset_index(drop=True)
    historicalMu_ = historicalMu_.reset_index(drop=True)
    historicalSigma_ = historicalSigma_.reset_index(drop=True)
    historicalPi_ = historicalPi_.reset_index(drop=True)
    historicalAlpha_ = historicalAlpha_.reset_index(drop=True)
    return historicalTrajectories, historicalProb_, historicalMu_, historicalSigma_, historicalPi_, historicalAlpha_
        
def selectTrajectories(retrievedTrajectories, historicalTrajectories, solution):
    with open(retrievedTrajectories, mode = 'w') as f:
        for i in trange(len(solution)):
            historicalTrajectories[solution[i]*60:(solution[i]+1)*60].to_csv(f, header = None, index = False)
    return 0

def retrieval(scoreFile, targetProb_, historicalProb_, targetNum, retrievedTrajectories, historicalTrajectories, paraNum):
    tree = KDTree(historicalProb_)
    solution = []
    wf = open(scoreFile, mode='w')
    for i in trange(len(targetProb_)):
        nearest_dist, nearest_ind = tree.query(targetProb_[i].reshape((1,paraNum*embedding_dim)), k=targetNum)
        meanLoss = nearest_dist[0].mean()
        wf.write(str(i) + ',' + str(meanLoss) + '\n')
        solution += list(nearest_ind[0])
    selectTrajectories(retrievedTrajectories, historicalTrajectories, solution)
    wf.close()
    return 0


def main(args):
    BATCH_SIZE = 16
    path_ = '../results/{}/KDTree{}'.format(args.model, args.model)
    if not os.path.exists(path_):
        os.mkdir(path_)
    path_ = '../results/{}/KDTree{}/EMD/'.format(args.model, args.model)
    if not os.path.exists(path_):
        os.mkdir(path_)
    historicalData = '../data/Experiment/query/query_data_before_time/'
    targetData = '../data/Experiment/query/query_data_before_time/{}_{}.csv'.format(args.day, args.hour)
    para = {
        "AE": {
            'history':{
                'historicalProb':'../results/AE/Index/Query/prob/'
            },
            'target':{
                'targetProb' : '../results/AE/Index/Query/prob/prob_8_17.csv'
            }
        },
        "VAE": {
            'history':{
                'historicalMu':'../results/VAE/Index/Query/mu/',
                'historicalSigma': '../results/VAE/Index/Query/sigma/'
            },
            'target':{
                'targetMu' : '../results/VAE/Index/Query/mu/mu_8_17.csv',
                'targetSigma' : '../results/VAE/Index/Query/sigma/sigma_8_17.csv'
            }

        },
        "VAE_nvib": {
            'history':{
                'historicalMu':'../results/VAE_nvib/Index/Query/mu/',
                'historicalSigma': '../results/VAE_nvib/Index/Query/sigma/',
                'historicalPi': '../results/VAE_nvib/Index/Query/pi/',
                'historicalAlpha': '../results/VAE_nvib/Index/Query/alpha/'
            },
            'target':{
                 'targetMu' : '../results/VAE_nvib/Index/Query/mu/mu_8_17.csv',
                'targetSigma' : '../results/VAE_nvib/Index/Query/sigma/sigma_8_17.csv',
                'targetPi' : '../results/VAE_nvib/Index/Query/pi/pi_8_17.csv',
                'targetAlpha' : '../results/VAE_nvib/Index/Query/alpha/alpha_8_17.csv'
            }
        }

    }
    

    groundTruthPath = '../data/Experiment/groundTruth/groundTruth_{}.csv'.format(args.day)
    scoreFile = '../results/{}/KDTree{}/EMD/meanLoss.csv'.format(args.model, args.model)
    targetTrajectories, targetDict = loadData(targetData, BATCH_SIZE, **(para[args.model]['target']))
    targerLen = len(targetTrajectories)
    groundTruthLen = len(pd.read_csv(groundTruthPath, header = None))
    print('target length: {}'.format(targerLen))
    print('ground truth length: {}'.format(groundTruthLen))
    targetProb_ = pd.concat(targetDict.values(), axis=1)
    print(targetProb_.shape)
    targetNum = int(groundTruthLen / targerLen)
    print('targetNum: {}'.format(targetNum))
    historicalTrajectories, historicalProb_, historicalMu_, historicalSigma_, historicalPi_, historicalAlpha_ = loadHistoricalData(historicalData, BATCH_SIZE, 
                                                                                                                  targetData, history=6, **(para[args.model]['history']))
    historicalProb_ = pd.concat([historicalProb_, historicalMu_, historicalSigma_, historicalPi_, historicalAlpha_], axis=1)
    historicalProb_ = historicalProb_.values
    print(historicalProb_.shape)
    retrievedTrajectories = '../results/{}/KDTree{}/EMD/retrievedTrajectories.csv'.format(args.model, args.model)
    # retrieval(scoreFile, targetProb_, historicalProb_, targetNum, retrievedTrajectories, historicalTrajectories, len(para[args.model]['history']))
    # retrievalLen = len(pd.read_csv(retrievedTrajectories, header = None))
    # print('retrieval length: {}'.format(retrievalLen))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model', type=str, default="AE", help='MODEL', choices=["AE", "VAE", "VAE_nvib"], required=True)
    parser.add_argument('-d','--day', type=int, default=2, help='day', required=True)
    parser.add_argument('-hour','--hour', type=int, default=0, help='hour', required=True)
    args = parser.parse_args()
    main(args)
