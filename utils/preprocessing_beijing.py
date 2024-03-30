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
        data = pd.read_csv(f, names=['id', 'timestamp', 'lon', 'lat'])
        data.to_csv('data/beijing/beijing.csv', mode='a', header=False, index=False)
    data = pd.read_csv('data/beijing/beijing.csv')
    data.columns = ['id', 'timestamp', 'lon', 'lat']
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
        tt = range(0, Config.traj_len, 1)
        lon = np.interp(tt, range(0, len(lon)), lon)
        lat = np.interp(tt, range(0, len(lat)), lat)
        res = [[lon[i], lat[i]] for i in range(Config.traj_len)]
        return res
    
def clean_and_output_data():
    _time = time.time()
    dfraw = pd.read_csv(Config.dataset_folder + 'beijing.csv') 
    dfraw = dfraw.sort_values(by=['id', 'timestamp'], ignore_index=True)
    
    # traj_segmented
    dfraw['time'] = pd.to_datetime(dfraw['timestamp'])
    dfraw['date'] = dfraw['time'].dt.date
    dfraw['hour'] = dfraw['time'].dt.hour
    dfraw['segment'] = (dfraw['id'] != dfraw['id'].shift()) | (dfraw['date'] != dfraw['date'].shift()) | (dfraw['time'] - dfraw['time'].shift() > pd.Timedelta(hours=1))
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
            
    
    