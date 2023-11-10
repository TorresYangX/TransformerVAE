# renumber trajctories in a given file

import numpy as np
import pandas as pd

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


if __name__ == '__main__':
    filename = '../small_results/VAE_transformer/KDTreeVAE_transformer/EMD/retrievedTrajectories.csv'
    OutputFile = '../small_results/VAE_transformer/KDTreeVAE_transformer/EMD/retrievedTrajectories_RN.csv'
    renumberTrajectories(filename, OutputFile)