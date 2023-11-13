import numpy as np
import pandas as pd
from tqdm import trange
from pyemd import emd
import argparse


def normalize(targetTrajectories):
    targetX = targetTrajectories[['latitude', 'longitude']]
    targetX.loc[:, 'latitude'] = (targetX.loc[:, 'latitude'] - 39.9) / 0.3
    targetX.loc[:, 'longitude'] = (targetX.loc[:, 'longitude'] - 116.4) / 0.4
    targetX = targetX.values.reshape(-1, 60, 2)
    return targetX

def cityEMD(targetTrajectories, reterievedTrajectories_, method, NLAT=6, NLON=8):
    targetX = normalize(targetTrajectories)
    retrievedY = normalize(reterievedTrajectories_)
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
    np.save('../results/{}/KDTree{}/EMD/emd_.npy'.format(method, method), emd_)
    return 0




def main(args):
    retrievedTrajectories = '../results/{}/KDTree{}/EMD/retrievedTrajectories.csv'.format(args.MODEL, args.MODEL)
    targetTrajectories = '../results/{}/KDTree{}/EMD/queryTrajectories.csv'.format(args.MODEL, args.MODEL)
    retrievedTrajectories_ = pd.read_csv(retrievedTrajectories, header = None)
    targetTrajectories = pd.read_csv(targetTrajectories, header = None)
    retrievedTrajectories_.columns = ['time', 'longitude', 'latitude', 'id']
    targetTrajectories.columns = ['time', 'longitude', 'latitude', 'id']
    {
        'cityEMD': cityEMD
    }(targetTrajectories, retrievedTrajectories_, args.MODEL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mo', '--MODEL', type=str, default='VAE', help='model name', choices=["AE", "VAE", "VAE_nvib", "LCSS", "EDR", "EDwP"], required=True)
    parser.add_argument('-md', '--METHOD', type=str, default='cityEMD', help='method', choices=["cityEMD"], required=True)
    args = parser.parse_args()
    main(args)
