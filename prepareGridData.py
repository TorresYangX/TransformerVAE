#imput: ../data/data_before_time
#output: ../data/grid_data

from math import sin, cos, sqrt, atan2, radians, floor
from prepareTrainingData import findfile
import os
import numpy as np
import pandas as pd

lat1 = 39.6
lat2 = 40.2
lon1 = 116
lon2 = 116.8
grid_num = 50

grid_size = max(lon2 - lon1, lat2 - lat1) / grid_num

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


def grid(data, grid_num):
    grid_data = []
    grid_size = max(lon2 - lon1, lat2 - lat1) / grid_num
    if grid_size == 0:
        return None
    for cnt in range(len(data) // 60):
        for row in data[cnt]:
            lon = float(row[0])
            lat = float(row[1])
            grid_id = min(floor((lon - lon1) / grid_size), grid_num - 1) * grid_num + min(
                floor((lat - lat1) / grid_size), grid_num - 1)
            grid_data.append([grid_id])
    return grid_data

def distancecalculate(data, hour):
    id_set = data.id.drop_duplicates()
    id_set = id_set.reset_index(drop = True)
    data_length = len(id_set)
    trajectory_length = 60 #length of trajectory
    dimension = 3#[grid_num, distance, hour]
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
            grid_id = min(floor((temp.longitude[i] - lon1) / grid_size), grid_num - 1) * grid_num + min(floor((temp.latitude[i] - lat1) / grid_size), grid_num - 1)
            temp.loc[i, 'grid_id'] = grid_id
                
        output_[j, :, 0] = temp.grid_id
        output_[j, :, 1] = temp.distance
        output_[j, :, 2] = [hour] * trajectory_length
    return output_

def prepareGridData(filePath, file, outputFilePath):
    data, hour = loadseperatedData(filePath, file)
    output_ = distancecalculate(data, hour)
    if not os.path.exists(outputFilePath):
        os.makedirs(outputFilePath)   
    filename = file.split('.')[0] + '.npy'
    np.save(outputFilePath + filename, output_)
    return 0

if __name__ == '__main__':
    filePath = '../small_data/query_data_before_time/'
    outputFilePath = '../small_data/queryGridData/'
    filelist = findfile(filePath)
    # [prepareGridData(filePath, file, outputFilePath) for file in filelist]
    for file in filelist:
        print(file)
        prepareGridData(filePath, file, outputFilePath)
    print('Done!')
            