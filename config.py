import os
import random
import torch
import numpy
from pandas import Timestamp

def set_seed(seed = -1):
    if seed == -1:
        return
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    debug = True
    dumpfile_uniqueid = ''
    seed = 2000
    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root_dir = os.path.abspath(__file__)[:-10] # dont use os.getcwd()
    checkpoint_dir = root_dir + '/exp/snapshots'
    
    # ================== DATASET ==================
    min_lon = 0.0
    min_lat = 0.0
    max_lon = 0.0
    max_lat = 0.0
    max_traj_len = 200
    min_traj_len = 20
    start_time = ''
    end_time = ''
    grid_file = ''
    lonlat_file = ''
    
    grid_num = 50
    
    
    # ================== NVAE ==================
    vocab_size = grid_num * grid_num + 2 #V
    Batch_size = 16 #B
    embedding_dim = 16 # H
    PRIOR_MU = 0
    PRIOR_VAR = 1
    PRIOR_ALPHA = 1
    KAPPA = 1
    DELTA = 1
    KL_GAUSSIAN_LAMBDA = 0.001
    KL_DIRICHLET_LAMBDA = 1
    KL_ANNEALING_GAUSSIAN = "constant"
    KL_ANNEALING_DIRICHLET = "constant"
    dropout = 0.1
    learning_rate = 0.001
    MAX_EPOCH = 500
    ACCUMULATION_STEPS = 1
    
    @classmethod
    def update(cls, dic: dict):
        for k, v in dic.items():
            if k in cls.__dict__:
                assert type(getattr(Config, k)) == type(v)
            setattr(Config, k, v)
        cls.post_value_updates()


    @classmethod
    def post_value_updates(cls):
        if 'porto' == cls.dataset:
            cls.dataset_prefix = 'porto'
            cls.min_lon = -8.7005
            cls.min_lat = 41.1001
            cls.max_lon = -8.5192
            cls.max_lat = 41.2086
            cls.start_time = Timestamp('2013-07-01 00:00:00')
            cls.end_time = Timestamp('2013-07-31 23:59:59')
        else:
            pass
        
        cls.dataset_file = cls.root_dir + '/data/' + cls.dataset_prefix
        cls.grid_file = cls.dataset_file + '_grid.pkl'
        cls.lonlat_file = cls.dataset_file + '_lonlat.pkl'

        set_seed(cls.seed)
         
    @classmethod
    def to_str(cls): # __str__, self
        dic = cls.__dict__.copy()
        lst = list(filter( \
                        lambda p: (not p[0].startswith('__')) and type(p[1]) != classmethod, \
                        dic.items() \
                        ))
        return '\n'.join([str(k) + ' = ' + str(v) for k, v in lst])