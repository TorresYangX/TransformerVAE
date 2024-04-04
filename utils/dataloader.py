import os
import time
import torch
import pandas as pd
import pickle as pickle
from config import Config
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
    
    train_idx = (int(l*0), int(l*0.8))
    eval_idx = (int(l*0.8), int(l*0.9))
    test_idx = (int(l*0.9), int(l*1.0))
    
    _train = TrajDataset(trajs[train_idx[0]:train_idx[1]])
    _eval = TrajDataset(trajs[eval_idx[0]: eval_idx[1]])
    _test = TrajDataset(trajs[test_idx[0]: test_idx[1]])
    
    logging.info('[Load traj dataset] END. @={:.0f}, #={}({}/{}/{})' \
                .format(time.time() - _time, l, len(_train), len(_eval), len(_test)))
    return _train, _eval, _test
    
class TrajDataset(Dataset):
    def __init__(self, data):
        # data: DataFrame
        self.data = data

    def __getitem__(self, index):
        return torch.tensor(self.data.loc[index])

    def __len__(self):
        return self.data.shape[0]