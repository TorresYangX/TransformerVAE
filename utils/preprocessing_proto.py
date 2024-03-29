import sys
sys.path.append('..')
import logging
logging.getLogger().setLevel(logging.INFO)


import time
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
    
    return

def generate_grid_data():
    def convert_multi_row(row):
        # convert to time, grid_id, taxi_id
        return [[pd.to_datetime(row['timestamp']), int((coord[0] - Config.min_lon) / Config.grid_size) * Config.grid_num + int((coord[1] - Config.min_lat) / Config.grid_size), row['TAXI_ID']] for coord in row['wgs_seq']]
    
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
    
    return
    

    
    


# class DataLoader:
#     def __init__(self, dataPath, rawTimeDataPath, girdDataPath):
#         self.dataPath = dataPath
#         self.rawTimeDataPath = rawTimeDataPath
#         self.girdDataPath = girdDataPath
        
#     def interpolate(self, x):
#         if len(x) == 60:
#             return x
#         else:
#             lon = [i[0] for i in x]
#             lat = [i[1] for i in x]
#             tt = range(0, 60, 1)
#             lon = np.interp(tt, range(0, len(lon)), lon)
#             lat = np.interp(tt, range(0, len(lat)), lat)
#             res = [[lon[i], lat[i]] for i in range(60)]
#             return res
        
#     def preprocess(self):
#         originData = pd.read_csv(self.dataPath)
#         data = originData[originData['MISSING_DATA'] == False]
#         data = data[['TAXI_ID', 'POLYLINE', 'TIMESTAMP']]
#         data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], unit='s')
#         data = data[(data['TIMESTAMP'] >= '2013-07-01 00:00:00') & (data['TIMESTAMP'] <= '2013-07-31 23:59:59')]
#         data['POLYLINE'] = data['POLYLINE'].apply(lambda x: eval(x))
#         data = data[data['POLYLINE'].apply(lambda x: len(x) > 60)]
#         # the range of latitude and longitude are 41.04-41.24 and -8.7--8.5
#         data = data[data['POLYLINE'].apply(lambda x: min([i[1] for i in x]) > 41.04 and max([i[1] for i in x]) < 41.24 and min([i[0] for i in x]) > -8.7 and max([i[0] for i in x]) < -8.5)]
        
        
#         if not os.path.exists(self.rawTimeDataPath):
#             os.makedirs(self.rawTimeDataPath)
#         for i in tqdm(range(1, 32)):
#             for j in range(24):
#                 timeStart = pd.to_datetime('2013-07-{} {}:00:00'.format(i, j))
#                 timeEnd = pd.to_datetime('2013-07-{} {}:59:59'.format(i, j))
#                 tempData = data[(data['TIMESTAMP'] >= timeStart) & (data['TIMESTAMP'] <= timeEnd)]
#                 # use interpolate to get the trajectory length to 60
#                 tempData.loc[:, 'POLYLINE'] = tempData['POLYLINE'].apply(lambda x: self.interpolate(x))
#                 tempData = tempData[tempData['POLYLINE'].apply(lambda x: len(x) == 60)]
#                 tempData = tempData.reset_index(drop=True)
#                 # when saving, do not save the column name and index
#                 with open(self.rawTimeDataPath + '{}_{}.csv'.format(i, j), mode='w') as f:
#                     tempData.to_csv(f, header = None, index = False)
    
#     def toGridData(self):
#         grid_num = 50
#         grid_size = 0.004
#         for i in tqdm(range(1, 32)):
#             for j in range(24):
#                 data = pd.read_csv(self.rawTimeDataPath + '{}_{}.csv'.format(i, j))
#                 data['POLYLINE'] = data['POLYLINE'].apply(lambda x: eval(x))
#                 data['GRID_ID'] = data['POLYLINE'].apply(lambda x: [int((i[0] + 8.7) / grid_size) * grid_num + int((i[1] - 41.04) / grid_size) for i in x])
#                 if not os.path.exists(self.girdDataPath):
#                     os.makedirs(self.girdDataPath)
#                 # concatenate the GRID_ID and TIMESTAMP, TAXI_ID, make it a 3D array, and save it as a .npy file
#                 output = np.zeros((len(data), 60, 3))
#                 for k in range(len(data)):
#                     output[k, :, 0] = data['GRID_ID'][k]
#                     output[k, :, 1] = [0] * 60
#                     output[k, :, 2] = [j] * 60
#                 np.save(self.girdDataPath + '{}_{}.npy'.format(i, j), output)

#     def toTimeData(self, timeDataPath):
#         # convert the format of rawtime data(id, polyline, timestamp) to (time, longitude, latitude, id), 1 row in rawtime data to 60 rows in time data, save as .csv file
#         if not os.path.exists(timeDataPath):
#             os.makedirs(timeDataPath)
#         for i in tqdm(range(1, 32)):
#             for j in range(24):
#                 data = pd.read_csv(self.rawTimeDataPath + '{}_{}.csv'.format(i, j), header=None)
#                 data['POLYLINE'] = data[1].apply(lambda x: eval(x))
#                 data['TIMESTAMP'] = data[2]
#                 data = data.drop([1, 2], axis=1)
#                 output = []
#                 for k in range(len(data)):
#                     for l in range(60):
#                         output.append([pd.to_datetime(data['TIMESTAMP'][k]) + pd.Timedelta(minutes=l), data['POLYLINE'][k][l][0], data['POLYLINE'][k][l][1], data[0][k]])
#                 output = pd.DataFrame(output)
#                 with open(timeDataPath + '{}_{}.csv'.format(i, j), mode='w') as f:
#                     output.to_csv(f, header = None, index = False)

            
                
        
if __name__ == '__main__':
    # dataPath = '../data/Porto/Origin.csv'
    # rawTimeDataPath = '../data/Porto/rawTimeData/'
    # TimeDataPath = '../data/Porto/timeData/'
    # gridDataPath = '../data/Porto/gridData/'
    # loader = DataLoader(dataPath, rawTimeDataPath,gridDataPath)
    # # loader.preprocess()
    # # loader.toGridData()
    # loader.toTimeData(TimeDataPath)
    clean_and_output_data()
                





