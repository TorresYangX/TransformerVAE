import os
import time
import logging  
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import Config
from ast import literal_eval

def prepare_csv():
    folder = '../data/Geolife/seperatedData/'
    files = os.listdir(folder)
    for file in tqdm(files):
        f = open(folder + file, 'r')
        data = pd.read_csv(f, names=['TAXI_ID', 'timestamp', 'lon', 'lat'])
        data.to_csv('data/beijing/beijing.csv', mode='a', header=False, index=False)
    data = pd.read_csv('data/beijing/beijing.csv')
    data.columns = ['TAXI_ID', 'timestamp', 'lon', 'lat']
    data.to_csv('data/beijing/beijing.csv', index=False)

def inrange(lon, lat):
    if lon <= Config.min_lon or lon >= Config.max_lon \
            or lat <= Config.min_lat or lat >= Config.max_lat:
        return False
    return True

def interpolate(x):
    if len(x) == Config.traj_len:
        return x
    else:
        lon = [i[0] for i in x]
        lat = [i[1] for i in x]
        tt = np.linspace(0, len(lon) - 1, Config.traj_len)
        lon_interp = np.interp(tt, np.arange(len(lon)), lon)
        lat_interp = np.interp(tt, np.arange(len(lat)), lat)
        res = [[lon_interp[i], lat_interp[i]] for i in range(Config.traj_len)]
        return res
    
def clean_and_output_data():
    _time = time.time()
    dfraw = pd.read_csv(Config.dataset_folder + 'beijing.csv') 
    dfraw = dfraw.sort_values(by=['TAXI_ID', 'timestamp'], ignore_index=True)
    
    # traj_segmented
    dfraw['time'] = pd.to_datetime(dfraw['timestamp'])
    dfraw['date'] = dfraw['time'].dt.date
    dfraw['hour'] = dfraw['time'].dt.hour
    dfraw['segment'] = (dfraw['TAXI_ID'] != dfraw['TAXI_ID'].shift()) | (dfraw['date'] != dfraw['date'].shift()) | (dfraw['time'] - dfraw['time'].shift() > pd.Timedelta(hours=1))
    dfraw['segment_id'] = dfraw['segment'].cumsum()
    df_wgs_seq = dfraw.groupby('segment_id').apply(lambda x: x[['lon', 'lat']].values.tolist()).reset_index()
    df_wgs_seq.columns = ['segment_id', 'wgs_seq']
    df_wgs_seq = df_wgs_seq.drop_duplicates(subset='segment_id', keep='first')
    dfraw = dfraw.merge(df_wgs_seq, on='segment_id', how='left')
    dfraw = dfraw.drop_duplicates(subset='segment_id', keep='first')
    dfraw = dfraw.reset_index(drop=True)
    
    dfraw.drop(['lon','lat', 'time', 'date', 'hour', 'segment', 'segment_id'], axis=1, inplace=True)
    dfraw.to_pickle(Config.dataset_file)
    logging.info('Preprocessed-traj. #traj={}'.format(dfraw.shape[0]))
    logging.info('Preprocess end. @={:.0f}'.format(time.time() - _time))
    
    # length requirement
    dfraw = pd.read_pickle(Config.dataset_file)
    dfraw['trajlen'] = dfraw.wgs_seq.apply(lambda traj: len(traj))
    dfraw = dfraw[(dfraw.trajlen >= Config.min_traj_len) & (dfraw.trajlen <= Config.max_traj_len)]
    logging.info('Preprocessed-rm length. #traj={}'.format(dfraw.shape[0]))
    
    # range requirement
    dfraw['inrange'] = dfraw.wgs_seq.map(lambda traj: sum([inrange(p[0], p[1]) for p in traj]) == len(traj) ) # True: valid
    dfraw = dfraw[dfraw.inrange == True]
    logging.info('Preprocessed-rm range. #traj={}'.format(dfraw.shape[0]))
    
    dfraw.to_pickle(Config.dataset_file)
    logging.info('Preprocess end. @={:.0f}'.format(time.time() - _time))
    return


def generate_lonlat_data():
    def convert_multi_row(row):
        return [[pd.to_datetime(row['timestamp']), coord[0], coord[1], row['TAXI_ID']] for coord in row['wgs_seq']]

    _time = time.time()
    dfraw = pd.read_pickle(Config.dataset_file)
    dfraw = dfraw[['TAXI_ID', 'wgs_seq', 'timestamp']]
    dfraw = dfraw.sort_values(by=['timestamp'], ignore_index=True)
    
    # interp to 60 length using nn.interp
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
    dfraw['timestamp'] = pd.to_datetime(dfraw['timestamp'])
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
    
    return
            
    
    