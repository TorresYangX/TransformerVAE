import os
import time
import torch
import pandas as pd
import pickle as pickle
from torch.utils.data import Dataset

import logging
logging.getLogger().setLevel(logging.INFO)

def read_traj_dataset(file_path):
    logging.info('[Load traj dataset] START.')
    _time = time.time()
    trajs = pd.read_pickle(file_path)
    trajs.columns = ['TAXI_ID', 'wgs_seq', 'timestamp', 'grid_seq']
    trajs = trajs['grid_seq']
    
    l = trajs.shape[0]
    logging.info('[traj dataset] Number of trajectories: {}'.format(l))
    
    dataset = TrajDataset(trajs)
    
    logging.info('[Load traj dataset] END. @={:.0f}' \
                .format(time.time() - _time, l))
    return dataset

    
    
class TrajDataset(Dataset):
    def __init__(self, data):
        # data: DataFrame
        self.data = data

    def __getitem__(self, index):
        return torch.tensor(self.data.loc[index])

    def __len__(self):
        return self.data.shape[0]