import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import os
from pyemd import emd
from tqdm import trange
import argparse

embedding_dim = 16
trajectory_length = 60


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
                print('something wrong!')
            output[key] = prob_
        elif key == 'Mu':
            mu_ = pd.read_csv(kwargs[key]+"mu_{}_{}.csv".format(day, hour), header = None)
            if int(len(trajectories)/trajectory_length) != len(mu_):
                print('something wrong!')
            output[key] = mu_
        elif key == 'Sigma':
            sigma_ = pd.read_csv(kwargs[key]+"sigma_{}_{}.csv".format(day, hour), header = None)
            if int(len(trajectories)/trajectory_length) != len(sigma_):
                print('something wrong!')
            output[key] = sigma_
        elif key == 'Pi':
            pi_ = pd.read_csv(kwargs[key]+"pi_{}_{}.csv".format(day, hour), header = None)
            if int(len(trajectories)/trajectory_length) != len(pi_):
                print('something wrong!')
            output[key] = pi_
        elif key == 'Alpha':
            alpha_ = pd.read_csv(kwargs[key]+"alpha_{}_{}.csv".format(day, hour), header = None)
            if int(len(trajectories)/trajectory_length) != len(alpha_):
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
        
def selectTrajectories(retrievedTrajectories, historicalTrajectories, solution):
    print('historicalTrajectories length: {}'.format(len(historicalTrajectories)))
    with open(retrievedTrajectories, mode = 'w') as f:
        for i in trange(len(solution)):
            historicalTrajectories[solution[i]*trajectory_length:(solution[i]+1)*trajectory_length
                                   ].to_csv(f, header = None, index = False)
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
    path_ = '../results/{}/{}/KDTree{}'.format(args.dataset,args.model, args.model)
    if not os.path.exists(path_):
        os.mkdir(path_)
    path_ = '../results/{}/{}/KDTree{}/EMD/'.format(args.dataset,args.model, args.model)
    if not os.path.exists(path_):
        os.mkdir(path_)
    if args.dataset == "Geolife":
        historicalData = '../data/Geolife/Experiment/experiment_data_before_time/'
        targetData = '../data/Geolife/Experiment/experiment_data_before_time/'
        groundTruthPath = '../data/Geolife/Experiment/groundTruth/groundTruth_{}.csv'.format(args.day)
    else:
        historicalData = '../data/Porto/timeData/'
        targetData = '../data/Porto/timeData/'
        groundTruthPath = '../data/Porto/groundTruth/groundTruth_{}.csv'.format(args.day)
    para = {
        "AE": {
            'Prob':'../results/{}/AE/Index/prob/'.format(args.dataset)
        },
        "VAE": {
            'Mu':'../results/{}/VAE/Index/mu/'.format(args.dataset),
            'Sigma': '../results/{}/VAE/Index/sigma/'.format(args.dataset)
        },
        "NVAE": {
            'Mu':'../results/{}/NVAE/Index/mu/'.format(args.dataset),
            'Sigma': '../results/{}/NVAE/Index/sigma/'.format(args.dataset),
            'Pi': '../results/{}/NVAE/Index/pi/'.format(args.dataset),
            'Alpha': '../results/{}/NVAE/Index/alpha/'.format(args.dataset)
        },
        "Transformer": {
            'Prob':'../results/{}/Transformer/Index/prob/'.format(args.dataset)
        },
        "t2vec": {
            'Prob':'../results/{}/AE/Index/prob/'.format(args.dataset)
        }
    }
    scoreFile = '../results/{}/{}/KDTree{}/EMD/meanLoss.csv'.format(args.dataset, args.model, args.model)
    targetTrajectories, targetDict = loadData(targetData, BATCH_SIZE, args.day, args.hour, **(para[args.model]))
    targerLen = len(targetTrajectories)
    groundTruthLen = len(pd.read_csv(groundTruthPath, header = None))
    print('target length: {}'.format(targerLen))
    print('ground truth length: {}'.format(groundTruthLen))
    targetProb_ = pd.concat(targetDict.values(), axis=1)
    targetProb_ = targetProb_.values
    targetNum = int(groundTruthLen / targerLen)
    print('targetNum: {}'.format(targetNum))
    historicalTrajectories, historicalProb_, historicalMu_, historicalSigma_, historicalPi_, historicalAlpha_ = loadHistoricalData(historicalData, BATCH_SIZE, 
                                                                                                                  args.day, args.hour, 
                                                                                                                  history=11, **(para[args.model]))
    historicalProb_ = pd.concat([historicalProb_, historicalMu_, historicalSigma_, historicalPi_, historicalAlpha_], axis=1)
    historicalProb_ = historicalProb_.values
    retrievedTrajectories = '../results/{}/{}/KDTree{}/EMD/retrievedTrajectories.csv'.format(args.dataset, args.model, args.model)
    retrieval(scoreFile, targetProb_, historicalProb_, targetNum, retrievedTrajectories, historicalTrajectories, len(para[args.model]))
    retrievalLen = len(pd.read_csv(retrievedTrajectories, header = None))
    print('retrieval length: {}'.format(retrievalLen))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model', type=str, default="t2vec", help='MODEL', choices=["AE", "VAE", "NVAE", "Transformer", "t2vec"], required=True)
    parser.add_argument("-d", "--dataset", type=str, default="Geolife", choices=["Geolife","Porto"] ,help="dataset", required=True)
    parser.add_argument('-day','--day', type=int, default=2, help='day', required=True)
    parser.add_argument('-hour','--hour', type=int, default=0, help='hour', required=True)
    args = parser.parse_args()
    main(args)
