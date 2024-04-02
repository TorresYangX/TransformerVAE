import os
import time
import logging  
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import Config
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
    
    
def generate_lonlat_data():
    
    def convert_multi_row(row):
        return [[pd.to_datetime(row['timestamp']), coord[0], coord[1], row['id']] for coord in row['wgs_seq']]
    
    _time = time.time()
    dfraw = pd.read_pickle(Config.dataset_file)
    dfraw = dfraw[['id', 'wgs_seq', 'timestamp']]
    dfraw['wgs_seq'] = dfraw['wgs_seq'].apply(lambda x: interpolate(x))
    dfraw = dfraw[dfraw['wgs_seq'].apply(lambda x: len(x) == 60)]
    dfraw = dfraw.reset_index(drop=True)
    dfraw.to_pickle(Config.intepolation_file)
    logging.info('Interpolated. #traj={}'.format(dfraw.shape[0]))
    
    # total traj data
    output = dfraw.apply(convert_multi_row, axis=1).explode().tolist()
    output = pd.DataFrame(output)
    output.to_pickle(Config.lonlat_total_file)
    logging.info('Saved lonlat_file. #traj={}'.format(output.shape[0]))
    
    # ground_data
    ground_data = dfraw[(dfraw['timestamp'] >= Config.ground_data_timerange[0]) & (dfraw['timestamp'] <= Config.ground_data_timerange[1])]
    __ground_data = ground_data.apply(convert_multi_row, axis=1).explode().tolist()
    __ground_data = pd.DataFrame(__ground_data)
    __ground_data.to_pickle(Config.lonlat_ground_file)
    logging.info('Saved ground_data_file. #traj={}'.format(ground_data.shape[0]))
    
    # sample test_data
    test_data = ground_data.sample(n=Config.test_data_num)
    __test_data = test_data.apply(convert_multi_row, axis=1).explode().tolist()
    __test_data = pd.DataFrame(__test_data)
    __test_data.to_pickle(Config.lonlat_test_file)
    logging.info('Saved test_data_file. #traj={}'.format(test_data.shape[0]))
    
    logging.info('Generate lonlat data end. @={:.0f}'.format(time.time() - _time))
    
    
def generate_grid_data():
    def convert_multi_row(row):
        # convert to time, grid_id, taxi_id
        return [[pd.to_datetime(row['timestamp']), int((coord[0] - Config.min_lon) / Config.grid_size) * Config.grid_num + int((coord[1] - Config.min_lat) / Config.grid_size), row['id']] for coord in row['wgs_seq']]
    
    _time = time.time()
    dfraw = pd.read_pickle(Config.intepolation_file)
    output = dfraw.apply(convert_multi_row, axis=1).explode().tolist()
    output = pd.DataFrame(output)
    output.to_pickle(Config.grid_total_file)
    logging.info('Saved grid_total_file. #traj={}'.format(output.shape[0]))
    
    ground_data = dfraw[(dfraw['timestamp'] >= Config.ground_data_timerange[0]) & (dfraw['timestamp'] <= Config.ground_data_timerange[1])]
    __ground_data = ground_data.apply(convert_multi_row, axis=1).explode().tolist()
    __ground_data = pd.DataFrame(__ground_data)
    __ground_data.to_pickle(Config.grid_ground_file)
    logging.info('Saved grid_ground_file. #traj={}'.format(ground_data.shape[0]))
    
    test_data = ground_data.sample(n=Config.test_data_num)
    __test_data = test_data.apply(convert_multi_row, axis=1).explode().tolist()
    __test_data = pd.DataFrame(__test_data)
    __test_data.to_pickle(Config.grid_test_file)
    logging.info('Saved grid_test_file. #traj={}'.format(test_data.shape[0]))
    
    logging.info('Generate grid data end. @={:.0f}'.format(time.time() - _time))
    
    return

    
    
        
        
        
        
    
    
    