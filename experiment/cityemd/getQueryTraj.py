'''
    get the query trajectory from origin data
'''

import numpy as np
import pandas as pd
import argparse

BATCH_SIZE = 16
trajectory_length = 60

def getQueryTraj(filename, OutputFile):
    trajectories = pd.read_csv(filename, header=None)
    trajectories.columns = ['time', 'longitude', 'latitude', 'id']
    resid = int(((len(trajectories) / trajectory_length) // BATCH_SIZE) * BATCH_SIZE * trajectory_length)
    trajectories = trajectories[:resid]
    totalID = int(trajectories.shape[0]/60)
    for i in range(totalID):
        for j in range(60):
            trajectories.loc[i*60+j, 'id'] = i
    with open(OutputFile, mode='w') as f:
        trajectories.to_csv(OutputFile, header=None, index=False)


def main(args):
    filepath = '../data/Experiment/experiment_data_before_time/8_17.csv'
    OutputFile = '../results/{}/KDTree{}/EMD/queryTrajectories_RN.csv'.format(args.model, args.model)
    getQueryTraj(filepath, OutputFile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='AE', help='model name', choices=['AE', 'VAE', 'NVAE', 'Transformer', 'LCSS', 'EDR', 'EDwP', 'DTW']),
    args = parser.parse_args()
    main(args)