import os
import time
import logging  
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset_config import DatasetConfig as Config
from ast import literal_eval


def inrange(lon, lat):
    if lon <= Config.min_lon or lon >= Config.max_lon \
            or lat <= Config.min_lat or lat >= Config.max_lat:
        return False
    return True

def interpolate(x):
    if len(x)/2 == Config.traj_len:
        return x
    else:
        lon = [x[i] for i in range(len(x)) if i % 2 == 0]
        lat = [x[i] for i in range(len(x)) if i % 2 == 1]
        tt = np.linspace(0, len(lon) - 1, Config.traj_len)
        lon_interp = np.interp(tt, np.arange(len(lon)), lon)
        lat_interp = np.interp(tt, np.arange(len(lat)), lat)
        res = [[lon_interp[i], lat_interp[i]] for i in range(Config.traj_len)]
        return res
    
def clean_and_output_data():
    origin_data_root = '../data/geolife/Data'
    
    def get_plt_filepath():
        plt_paths = []
        users = os.listdir(origin_data_root)
        for user in users:
            user_path = os.path.join(origin_data_root, user)
            file_path = os.path.join(user_path, 'Trajectory')
            traj_files = os.listdir(file_path)
            for traj_file in traj_files:
                # time requirement
                if '2008' in traj_file or '2009' in traj_file or '2010' in traj_file or '2011' in traj_file:
                    plt_paths.append(os.path.join(file_path, traj_file))
        return plt_paths
    
    def single_file_process(filepath):
        data = pd.read_csv(filepath, 
                           header=None,
                           skiprows=6,
                           names=['Latitude', 'Longitude', 'Not_Important1', 'Altitude', 'Not_Important2', 'Date', 'Time'])
    
        data = data[['Longitude', 'Latitude', 'Date', 'Time']]
        data['timestamp'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
        data=data[['Longitude', 'Latitude', 'timestamp']]
        data['is_moving'] = (data['Latitude'] != data['Latitude'].shift()) | (data['Longitude'] != data['Longitude'].shift())
        data = data[data['is_moving']]
        
        # range requirement
        data['inrange'] = data.apply(lambda row: inrange(row['Longitude'], row['Latitude']), axis=1)
        data = data[data.inrange == True]
        
        # split traj into two trajs if the time interval between two points is over 10 minutes
        data['time_diff'] = data['timestamp'].diff()
        data['id']=0
        mask=data['time_diff']>pd.Timedelta(minutes=10)
        data.loc[mask,'id']=1
        data['id']=data['id'].cumsum()
        
        # length requirement
        data['trajlen'] = data.groupby('id')['id'].transform('count')
        data = data[(data.trajlen >= Config.min_traj_len)]
        
        data = data[['Longitude', 'Latitude', 'timestamp', 'id']]
        data['wgs_seq'] = data[['Longitude', 'Latitude']].values.tolist()
        data = data.groupby('id').agg({'wgs_seq': 'sum', 'timestamp': 'first'}).reset_index()
        
        return data
    
    _time = time.time()
    plt_files = get_plt_filepath()
    dfraw = pd.DataFrame()
    for plt_file in tqdm(plt_files):
        data = single_file_process(plt_file)
        dfraw = pd.concat([dfraw, data])
    dfraw.to_pickle(Config.dataset_file)
    logging.info('Preprocess end. @={:.0f}'.format(time.time() - _time))
    
    
def generate_intepolation_data():
    dfraw = pd.read_pickle(Config.dataset_file)
    dfraw['wgs_seq'] = dfraw['wgs_seq'].apply(lambda x: interpolate(x))
    dfraw = dfraw[dfraw['wgs_seq'].apply(lambda x: len(x) == 60)]
    dfraw = dfraw.reset_index(drop=True)
    dfraw.to_pickle(Config.intepolation_file)
    logging.info('Interpolated. #traj={}'.format(dfraw.shape[0]))
    
def generate_lonlat_data():
    
    _time = time.time()
    dfraw = pd.read_pickle(Config.intepolation_file)
    
    # total traj data
    dfraw.to_pickle(Config.lonlat_total_file)
    logging.info('Saved lonlat_file. #traj={}'.format(dfraw.shape[0]))
    
    # ground_data
    ground_data = dfraw[(dfraw['timestamp'] >= Config.ground_data_timerange[0]) & (dfraw['timestamp'] <= Config.ground_data_timerange[1])]
    ground_data.to_pickle(Config.lonlat_ground_file)
    logging.info('Saved ground_data_file. #traj={}'.format(ground_data.shape[0]))
    
    # sample test_data
    test_data = ground_data.sample(n=Config.test_data_num)
    test_data.to_pickle(Config.lonlat_test_file)
    logging.info('Saved test_data_file. #traj={}'.format(test_data.shape[0]))
    
    logging.info('Generate lonlat data end. @={:.0f}'.format(time.time() - _time))
    
    
def generate_grid_data():
    
    def lonlat2grid(row):
        lon = [i[0] for i in row['wgs_seq']]
        lat = [i[1] for i in row['wgs_seq']]
        grid_list = [int((lon[i] - Config.min_lon) / Config.grid_size) * Config.grid_num + int((lat[i] - Config.min_lat) / Config.grid_size) for i in range(len(lon))]
        if min(grid_list) < 0 or max(grid_list) >= Config.grid_num * Config.grid_num:
            print('Error: grid_list out of range')
        return grid_list
        
        
    _time = time.time()
    # total traj data
    dfraw = pd.read_pickle(Config.lonlat_total_file)
    dfraw = dfraw[['id', 'wgs_seq', 'timestamp']]
    dfraw['grid_seq'] = dfraw.apply(lonlat2grid, axis=1)
    dfraw = dfraw.reset_index(drop=True)
    dfraw.to_pickle(Config.grid_total_file)
    logging.info('Saved grid_total_file. #traj={}'.format(dfraw.shape[0]))
    
    # ground_data
    ground_data  = pd.read_pickle(Config.lonlat_ground_file)
    ground_data = ground_data[['id', 'wgs_seq', 'timestamp']]
    ground_data['grid_seq'] = ground_data.apply(lonlat2grid, axis=1)
    ground_data = ground_data.reset_index(drop=True)
    ground_data.to_pickle(Config.grid_ground_file)
    logging.info('Saved grid_ground_file. #traj={}'.format(ground_data.shape[0]))
    
    # test_data
    test_data = pd.read_pickle(Config.lonlat_test_file)
    test_data = test_data[['id', 'wgs_seq', 'timestamp']]
    test_data['grid_seq'] = test_data.apply(lonlat2grid, axis=1)
    test_data = test_data.reset_index(drop=True)
    test_data.to_pickle(Config.grid_test_file)
    logging.info('Saved grid_test_file. #traj={}'.format(test_data.shape[0]))
    
    logging.info('Generate grid data end. @={:.0f}'.format(time.time() - _time))
    
    return

    
    
        
        
        
        
    
    
    