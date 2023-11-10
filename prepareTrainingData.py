### prepare training data from seperatedData

# inputPath = '../data/data_before_time/'
# outputPath = '../data/trainingData/'

import pandas as pd
import numpy as np
from math import sin, cos, sqrt, atan2, radians
import glob
import os
from tqdm import trange

def findfile(filePath):
    original_path = os.getcwd()
    os.chdir(filePath)
    filelist = []
    for i, file in enumerate(glob.glob('*')):
        filelist.append(file)
    os.chdir(original_path)
    return filelist

def loadseperatedData(filePath, file):
    hour = file.split('_')[1].split('.')[0]
    data = pd.read_csv(filePath + file, header = None)
    data.columns = ['time', 'longitude', 'latitude', 'id']
    return data, int(hour)

def herversine(lon1, lat1, lon2, lat2):
    R = 6373000
    
    lon1 = radians(lon1)
    lat1 = radians(lat1)
    lon2 = radians(lon2)
    lat2 = radians(lat2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    return distance

def distancecalculate(data, hour):
    id_set = data.id.drop_duplicates()
    id_set = id_set.reset_index(drop = True)
    data_length = len(id_set)
    trajectory_length = 60 #length of trajectory
    dimension = 4#[lon, lat, distance, hour]
    output_ = np.zeros((data_length, trajectory_length, dimension))
    for j in range(data_length):
        k = id_set[j]
        temp = data[data.id == k]
        temp = temp.reset_index(drop=True)
        for i in range(len(temp)):
            if i == 0:
                temp.loc[i, 'distance'] = 0
            else:
                temp.loc[i, 'distance'] = herversine(temp.longitude[i-1],temp.latitude[i-1],temp.longitude[i],temp.latitude[i])
        output_[j, :, 0] = temp.longitude
        output_[j, :, 1] = temp.latitude
        output_[j, :, 2] = temp.distance
        output_[j, :, 3] = [hour] * trajectory_length
    return output_

def prepareData(filePath, file, outputFilePath):
    data, hour = loadseperatedData(filePath, file)
    output_ = distancecalculate(data, hour)
    if not os.path.exists(outputFilePath):
        os.makedirs(outputFilePath)
    
    filename = file.split('.')[0] + '.npy'
    np.save(outputFilePath + filename, output_)
    return 0

if __name__ == '__main__':
    filePath = '../data/train_data_before_time/'
    outputFilePath = '../data/trainingData/'
    filelist = findfile(filePath)
    print('Start prepare training data')
    for i in trange(0, len(filelist)):
        prepareData(filePath, filelist[i], outputFilePath)
    print('Done!')
    filePath = '../data/query_data_before_time/'
    outputFilePath = '../data/queryData/'
    filelist = findfile(filePath)
    print('Start prepare query data')
    for i in trange(0, len(filelist)):
        prepareData(filePath, filelist[i], outputFilePath)
    print('Done!')

            
