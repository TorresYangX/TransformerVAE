# renumber trajctories in a given file

import numpy as np
import pandas as pd
import argparse

def renumberTrajectories(filename, OutputFile):
    # read in data
    df = pd.read_csv(filename, header=None)
    df.columns = ['time', 'longitude', 'latitude', 'id']
    # renumber trajectories
    totalID = int(df.shape[0]/60)
    for i in range(totalID):
        for j in range(60):
            df.loc[i*60+j, 'id'] = i
    # write out data
    with open(OutputFile, mode='w') as f:
        df.to_csv(OutputFile, header=None, index=False)


def main(args):
    filePath = '../results/{}/KDTree{}/EMD/retrievedTrajectories.csv'.format(args.model, args.model)
    OutputFile = '../results/{}/KDTree{}/EMD/retrievedTrajectories_RN.csv'.format(args.model, args.model)
    renumberTrajectories(filePath, OutputFile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='AE', help='model name', choices=['AE', 'VAE', 'NVAE', 'Transformer', 'LCSS', 'EDR', 'EDwP', 'DTW']),
    args = parser.parse_args()
    main(args)