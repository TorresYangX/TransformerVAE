import sys
sys.path.append('..')
import logging
logging.getLogger().setLevel(logging.INFO)


import time
import numpy as np
import pandas as pd
from config import Config
from ast import literal_eval

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
    dfraw = pd.read_csv(Config.dataset_folder + 'porto.csv') 
    # Columns:
    # 'TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID', 'TIMESTAMP', 'DAY_TYPE', 'MISSING_DATA', 'POLYLINE'
    dfraw = dfraw.rename(columns = {"POLYLINE": "wgs_seq"})
    
    dfraw = dfraw[dfraw.MISSING_DATA == False]
    
    # time requirement
    dfraw['timestamp'] = pd.to_datetime(dfraw['TIMESTAMP'], unit='s')
    dfraw = dfraw[(dfraw['timestamp'] >= Config.start_time) & (dfraw['timestamp'] <= Config.end_time)]
    logging.info('Preprocessed-rm time. #traj={}'.format(dfraw.shape[0]))
    
    # length requirement
    dfraw.wgs_seq = dfraw.wgs_seq.apply(literal_eval)
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

def generate_grid_data():
    def convert_multi_row(row):
        # convert to time, grid_id, taxi_id
        return [[pd.to_datetime(row['timestamp']), int((coord[0] - Config.min_lon) / Config.grid_size) * Config.grid_num + int((coord[1] - Config.min_lat) / Config.grid_size), row['TAXI_ID']] for coord in row['wgs_seq']]
    
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

            
        
                





