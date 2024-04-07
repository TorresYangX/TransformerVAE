import os
import torch
from pandas import Timestamp

class DatasetConfig:
    
    root_dir = os.path.abspath(__file__)[:-18]
    
    dataset = 'porto'
    
    dataset_folder = ''
    grid_folder = ''
    lonlat_folder = ''
    
    dataset_file = ''
    intepolation_file = ''
    
    lonlat_total_file = ''
    lonlat_ground_file = ''
    lonlat_test_file = ''
    
    grid_total_file = ''
    grid_ground_file = ''
    grid_test_file = ''
    
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
    grid_size = 0.0
    grid_num = 50
    
    @classmethod
    def update(cls, dic: dict):
        for k, v in dic.items():
            if k in cls.__dict__:
                assert type(getattr(DatasetConfig, k)) == type(v)
            setattr(DatasetConfig, k, v)
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
            
            cls.grid_size = max((cls.max_lat-cls.min_lat), (cls.max_lon-cls.min_lon))/cls.grid_num
            
        elif 'geolife' == cls.dataset:
            cls.dataset_prefix = 'geolife'
            cls.min_lon = 116.20
            cls.min_lat = 39.75
            cls.max_lon = 116.60
            cls.max_lat = 40.05
            cls.min_traj_len = 20
            cls.test_data_num = 50
            cls.start_time = Timestamp('2008-01-01 00:00:00')
            cls.end_time = Timestamp('2011-12-31 23:59:59')
            cls.ground_data_timerange = [Timestamp('2009-05-01 00:00:00'), Timestamp('2009-05-31 23:59:59')]
            
            cls.grid_size = max((cls.max_lat-cls.min_lat), (cls.max_lon-cls.min_lon))/cls.grid_num
            
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
         
    @classmethod
    def to_str(cls): # __str__, self
        dic = cls.__dict__.copy()
        lst = list(filter( \
                        lambda p: (not p[0].startswith('__')) and type(p[1]) != classmethod, \
                        dic.items() \
                        ))
        return '\n'.join([str(k) + ' = ' + str(v) for k, v in lst])