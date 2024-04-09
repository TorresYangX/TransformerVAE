import os
import psutil
import numpy as np
import pandas as pd
from pynvml import *
from model_config import ModelConfig

nvmlInit() # need initializztion here

def mean(x):
    if x == []:
        return 0.0
    return sum(x) / len(x)
    

def pkl2csv(pkl_file, csv_file):
    df = pd.read_pickle(pkl_file)
    df.to_csv(csv_file, index=False, header=None)
    
def get_real_ground(ground_file, test_file, _r_ground_file_path):
    test_file_len = pd.read_pickle(test_file).shape[0] - pd.read_pickle(test_file).shape[0] % ModelConfig.NVAE.BATCH_SIZE 
    ground_data = pd.read_pickle(ground_file)
    ground_data.columns = ['TAXI_ID', 'wgs_seq', 'timestamp']
    ground_data = ground_data.iloc[:-(ground_data.shape[0] % test_file_len)]
    ground_data = ground_data.reset_index(drop=True)
    ground_data.to_pickle(_r_ground_file_path)
    return 
    
    
    
class GPUInfo:

    _h = nvmlDeviceGetHandleByIndex(0)

    @classmethod
    def mem(cls):
        info = nvmlDeviceGetMemoryInfo(cls._h)
        return info.used // 1048576, info.total // 1048576 # in MB

class RAMInfo:
    @classmethod
    def mem(cls):
        return int(psutil.Process(os.getpid()).memory_info().rss / 1048576) # in MB
