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
    
    traj_len = 60
    grid_num = 50
    
    # ================== DATASET ==================
    min_lon = 0.0
    min_lat = 0.0
    max_lon = 0.0
    max_lat = 0.0
    max_traj_len = 200
    min_traj_len = 20
    start_time = ''
    end_time = ''
    test_data_num = 0
    ground_data_timerange = []
    
    
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
            cls.min_lon = -8.6705
            cls.min_lat = 41.0801
            cls.max_lon = -8.5205
            cls.max_lat = 41.2301
            cls.start_time = Timestamp('2013-07-01 00:00:00')
            cls.end_time = Timestamp('2013-07-31 23:59:59')
            cls.test_data_num = 50
            cls.ground_data_timerange = [Timestamp('2013-07-15 00:00:00'), Timestamp('2013-07-15 23:59:59')]
            
            cls.grid_size = (cls.max_lat-cls.min_lat)/cls.grid_num
            
        else:
            pass
        
        cls.dataset_folder = cls.root_dir + '/data/' + cls.dataset_prefix + '/'
        cls.grid_folder = cls.dataset_folder + 'grid/'
        cls.lonlat_folder = cls.dataset_folder + 'lonlat/'
        cls.dataset_file = cls.dataset_folder + cls.dataset_prefix + '.pkl'
        cls.intepolation_file = cls.dataset_folder + cls.dataset_prefix + '_interpolation.pkl'
        
        cls.lonlat_total_file = cls.lonlat_folder + cls.dataset_prefix + '_total.pkl'
        cls.lonlat_ground_file = cls.lonlat_folder + cls.dataset_prefix + '_ground_data.pkl'
        cls.lonlat_test_file = cls.lonlat_folder + cls.dataset_prefix + '_test_data.pkl'
        
        cls.grid_total_file = cls.grid_folder + cls.dataset_prefix + '_total.pkl'
        cls.grid_ground_file = cls.grid_folder + cls.dataset_prefix + '_ground_data.pkl'
        cls.grid_test_file = cls.grid_folder + cls.dataset_prefix + '_test_data.pkl'

        set_seed(cls.seed)
         
    @classmethod
    def to_str(cls): # __str__, self
        dic = cls.__dict__.copy()
        lst = list(filter( \
                        lambda p: (not p[0].startswith('__')) and type(p[1]) != classmethod, \
                        dic.items() \
                        ))
        return '\n'.join([str(k) + ' = ' + str(v) for k, v in lst])