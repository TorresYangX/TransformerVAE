import numpy as np
import pandas as pd
from tqdm import trange
from pyemd import emd
import argparse


def normalize(trajectories):
    targetX = trajectories[['latitude', 'longitude']]
    targetX.loc[:, 'latitude'] = (targetX.loc[:, 'latitude'] - 41.14) / 0.1
    targetX.loc[:, 'longitude'] = (targetX.loc[:, 'longitude'] + 8.6) / 0.1
    targetX = targetX.values.reshape(-1, 60, 2)
    return targetX

def cityEMD(groundTruth, reterievedTrajectories_, method, dataset, NLAT=32, NLON=32):
    targetX = normalize(groundTruth)
    print('targetX.shape: {}'.format(targetX.shape))
    retrievedY = normalize(reterievedTrajectories_)
    print('retrievedY.shape: {}'.format(retrievedY.shape))
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
    for kt in trange(59):
        for it in trange(NLAT*NLON):
            emd_[kt] += emd(flowReal[it, :, kt].copy(order='C'), flowRetrieved[it, :, kt].copy(order='C'), flowDistance)
    np.save('../results/{}/{}/KDTree{}/EMD/emd_.npy'.format(dataset, method, method), emd_)
    return 0




def main(args):
    retrievedTrajectories = '../results/{}/{}/KDTree{}/EMD/retrievedTrajectories.csv'.format(args.DATASET, args.MODEL, args.MODEL)
    groundTruth = '../data/Porto/groundTruth/groundTruth_12.csv'
    retrievedTrajectories_ = pd.read_csv(retrievedTrajectories, header = None)
    groundTruth = pd.read_csv(groundTruth, header = None)
    # delete last rows of groundTruth so that the length of groundTruth is the same as retrievedTrajectories
    groundTruth = groundTruth[:len(retrievedTrajectories_)]
    retrievedTrajectories_.columns = ['time', 'longitude', 'latitude', 'id']
    groundTruth.columns = ['time', 'longitude', 'latitude', 'id']
    cityEMD(groundTruth, retrievedTrajectories_, args.MODEL, args.DATASET)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--DATASET', type=str, default='Porto', help='dataset name', choices=["Geolife", "Porto"], required=True)
    parser.add_argument('-m', '--MODEL', type=str, default='t2vec', help='model name', choices=["AE", "VAE", "NVAE", "t2vec",
                                                                                              "Transformer", "LCSS", "EDR", 
                                                                                              "EDwP", "DTW"], required=True)
    args = parser.parse_args()
    main(args)
