'''
input: ../data/Porto/Origin.csv
output: ../data/Porto/Train/rawTimeData/
time range: start: 2013-07-01 00:00:00; end: 2013-07-31 23:59:59
origin trajectory min length: 60
trajectory length: 60
'''

import pandas as pd
import os
from tqdm import tqdm
import numpy as np

class DataLoader:
    def __init__(self, dataPath, outputPath):
        self.dataPath = dataPath
        self.outputPath = outputPath
        
    def preprocess(self):
        originData = pd.read_csv(self.dataPath)
        data = originData[originData['MISSING_DATA'] == False]
        data = data[['TAXI_ID', 'POLYLINE', 'TIMESTAMP']]
        data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], unit='s')
        data = data[(data['TIMESTAMP'] >= '2013-07-01 00:00:00') & (data['TIMESTAMP'] <= '2013-07-31 23:59:59')]
        data['POLYLINE'] = data['POLYLINE'].apply(lambda x: eval(x))
        data = data[data['POLYLINE'].apply(lambda x: len(x) > 60)]
        # save as tmp file
        data.to_csv('../data/Porto/tmp.csv', index=False)
        print(data.shape)
        
        if not os.path.exists(self.outputPath):
            os.makedirs(self.outputPath)
        for i in tqdm(range(1, 32)):
            for j in range(24):
                timeStart = pd.to_datetime('2013-07-{} {}:00:00'.format(i, j))
                timeEnd = pd.to_datetime('2013-07-{} {}:59:59'.format(i, j))
                tempData = data[(data['TIMESTAMP'] >= timeStart) & (data['TIMESTAMP'] <= timeEnd)]
                # use interpolate to get the trajectory length to 60
                tempData.loc[:, 'POLYLINE'] = tempData['POLYLINE'].apply(lambda x: self.interpolate(x))
                tempData = tempData[tempData['POLYLINE'].apply(lambda x: len(x) == 60)]
                tempData = tempData.reset_index(drop=True)
                tempData.to_csv(self.outputPath + '{}_{}.csv'.format(i, j), index=False)
                
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
        
if __name__ == '__main__':
    dataPath = '../data/Porto/Origin.csv'
    outputPath = '../data/Porto/Train/rawTimeData/'
    loader = DataLoader(dataPath, outputPath)
    loader.preprocess()
                





