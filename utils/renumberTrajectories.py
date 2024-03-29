import numpy as np
import pandas as pd
from config import Config

def renumberTrajectories(filename, OutputFile):
    # read in data
    df = pd.read_csv(filename, header=None)
    df.columns = ['time', 'longitude', 'latitude', 'id']
    # renumber trajectories
    totalID = int(df.shape[0]/Config.traj_len)
    for i in range(totalID):
        for j in range(Config.traj_len):
            df.loc[i*Config.traj_len+j, 'id'] = i
    # write out data
    with open(OutputFile, mode='w') as f:
        df.to_csv(OutputFile, header=None, index=False)