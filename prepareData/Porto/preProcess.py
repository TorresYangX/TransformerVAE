'''
input: ../data/Porto/Origin.csv
output: ../data/Porto/rawTimeData/; ../data/Porto/gridData/
time range: start: 2013-07-01 00:00:00; end: 2013-07-31 23:59:59
origin trajectory min length: 60
trajectory length: 60
latitude range: 41.04-41.24
longitude range: -8.7--8.5
'''

import pandas as pd
import os
from tqdm import tqdm
import numpy as np

class DataLoader:
    def __init__(self, dataPath, rawTimeDataPath, girdDataPath):
        self.dataPath = dataPath
        self.rawTimeDataPath = rawTimeDataPath
        self.girdDataPath = girdDataPath
        
    def interpolate(self, x):
        if len(x) == 60:
            return x
        else:
            lon = [i[0] for i in x]
            lat = [i[1] for i in x]
            tt = range(0, 60, 1)
            lon = np.interp(tt, range(0, len(lon)), lon)
            lat = np.interp(tt, range(0, len(lat)), lat)
            res = [[lon[i], lat[i]] for i in range(60)]
            return res
        
    def preprocess(self):
        originData = pd.read_csv(self.dataPath)
        data = originData[originData['MISSING_DATA'] == False]
        data = data[['TAXI_ID', 'POLYLINE', 'TIMESTAMP']]
        data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], unit='s')
        data = data[(data['TIMESTAMP'] >= '2013-07-01 00:00:00') & (data['TIMESTAMP'] <= '2013-07-31 23:59:59')]
        data['POLYLINE'] = data['POLYLINE'].apply(lambda x: eval(x))
        data = data[data['POLYLINE'].apply(lambda x: len(x) > 60)]
        # the range of latitude and longitude are 41.04-41.24 and -8.7--8.5
        data = data[data['POLYLINE'].apply(lambda x: min([i[1] for i in x]) > 41.04 and max([i[1] for i in x]) < 41.24 and min([i[0] for i in x]) > -8.7 and max([i[0] for i in x]) < -8.5)]
        
        
        if not os.path.exists(self.rawTimeDataPath):
            os.makedirs(self.rawTimeDataPath)
        for i in tqdm(range(1, 32)):
            for j in range(24):
                timeStart = pd.to_datetime('2013-07-{} {}:00:00'.format(i, j))
                timeEnd = pd.to_datetime('2013-07-{} {}:59:59'.format(i, j))
                tempData = data[(data['TIMESTAMP'] >= timeStart) & (data['TIMESTAMP'] <= timeEnd)]
                # use interpolate to get the trajectory length to 60
                tempData.loc[:, 'POLYLINE'] = tempData['POLYLINE'].apply(lambda x: self.interpolate(x))
                tempData = tempData[tempData['POLYLINE'].apply(lambda x: len(x) == 60)]
                tempData = tempData.reset_index(drop=True)
                # when saving, do not save the column name and index
                with open(self.rawTimeDataPath + '{}_{}.csv'.format(i, j), mode='w') as f:
                    tempData.to_csv(f, header = None, index = False)
    
    def toGridData(self):
        grid_num = 50
        grid_size = 0.004
        for i in tqdm(range(1, 32)):
            for j in range(24):
                data = pd.read_csv(self.rawTimeDataPath + '{}_{}.csv'.format(i, j))
                data['POLYLINE'] = data['POLYLINE'].apply(lambda x: eval(x))
                data['GRID_ID'] = data['POLYLINE'].apply(lambda x: [int((i[0] + 8.7) / grid_size) * grid_num + int((i[1] - 41.04) / grid_size) for i in x])
                if not os.path.exists(self.girdDataPath):
                    os.makedirs(self.girdDataPath)
                # concatenate the GRID_ID and TIMESTAMP, TAXI_ID, make it a 3D array, and save it as a .npy file
                output = np.zeros((len(data), 60, 3))
                for k in range(len(data)):
                    output[k, :, 0] = data['GRID_ID'][k]
                    output[k, :, 1] = [0] * 60
                    output[k, :, 2] = [j] * 60
                np.save(self.girdDataPath + '{}_{}.npy'.format(i, j), output)

    def toTimeData(self, timeDataPath):
        # convert the format of rawtime data(id, polyline, timestamp) to (time, longitude, latitude, id), 1 row in rawtime data to 60 rows in time data, save as .csv file
        if not os.path.exists(timeDataPath):
            os.makedirs(timeDataPath)
        for i in tqdm(range(1, 32)):
            for j in range(24):
                data = pd.read_csv(self.rawTimeDataPath + '{}_{}.csv'.format(i, j), header=None)
                data['POLYLINE'] = data[1].apply(lambda x: eval(x))
                data['TIMESTAMP'] = data[2]
                data = data.drop([1, 2], axis=1)
                output = []
                for k in range(len(data)):
                    for l in range(60):
                        output.append([pd.to_datetime(data['TIMESTAMP'][k]) + pd.Timedelta(minutes=l), data['POLYLINE'][k][l][0], data['POLYLINE'][k][l][1], data[0][k]])
                output = pd.DataFrame(output)
                with open(timeDataPath + '{}_{}.csv'.format(i, j), mode='w') as f:
                    output.to_csv(f, header = None, index = False)

            
                
        
if __name__ == '__main__':
    dataPath = '../data/Porto/Origin.csv'
    rawTimeDataPath = '../data/Porto/rawTimeData/'
    TimeDataPath = '../data/Porto/timeData/'
    gridDataPath = '../data/Porto/gridData/'
    loader = DataLoader(dataPath, rawTimeDataPath,gridDataPath)
    # loader.preprocess()
    # loader.toGridData()
    loader.toTimeData(TimeDataPath)
                





