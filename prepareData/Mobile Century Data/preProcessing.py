'''
input path: ../data/MobileCenturyData/Origin/NB_veh_files or ../data/MobileCenturyData/Origin/SB_veh_files
output path: ../data/MobileCenturyData/Train/ or ../data/MobileCenturyData/Experiment/
'''

import os
import numpy as np
import pandas as pd
from tqdm import trange
import datetime

class DataLoader(object):
    def __init__(self, args):
        self.inputPath = args['inputPath']
        self.startid = args['startid']
        self.endid = args['endid']
        self.timeInterval = args['timeInterval']

        
    def loadData(self):
        for id_num in trange(self.startid, self.endid):
            filePath_ = self.inputPath + '/veh_' + str(id_num) + '.csv'
            data_ = pd.read_csv(filePath_)
            data_.columns = ['timestep', 'latitude', 'longitude', 'postmile', 'speed']
            data_['timestep'] = pd.to_datetime(data_.timestep)
            print(data_)
            

    
    
if __name__ == '__main__':
    argments = {
        'inputPath': '../data/MobileCenturyData/Origin/NB_veh_files',
        'startid': 1,
        'endid': 2,
        'timeInterval': 5
    }
    dataLoader = DataLoader(argments)
    dataLoader.loadData()
            