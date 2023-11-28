#imput: ../data/data_before_time
#output: ../data/grid_data

from math import sin, cos, sqrt, atan2, radians, floor
from prepareTrainingData import findfile
import os
import numpy as np
import pandas as pd
from tqdm import trange
import argparse

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


def distancecalculate(data, hour, traj_length):
    id_set = data.id.drop_duplicates()
    id_set = id_set.reset_index(drop = True)
    data_length = len(id_set)
    dimension = 3#[grid_num, distance, hour]
    output_ = np.zeros((data_length, traj_length, dimension))
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
        output_[j, :, 2] = [hour] * traj_length
    return output_

def prepareGridData(filePath, file, outputFilePath, traj_length):
    data, hour = loadseperatedData(filePath, file)
    output_ = distancecalculate(data, hour, traj_length)
    if not os.path.exists(outputFilePath):
        os.makedirs(outputFilePath)   
    filename = file.split('.')[0] + '.npy'
    np.save(outputFilePath + filename, output_)
    return 0

def main(args):
    header1 = '../data/Train/'
    header2 = '../data/Experiment/'
    if args.model == 'train':
        header = header1
        if args.SSN:
            header = header + 'SSM_KNN/Database/'
            filePath = header + 'data_before_time/'
            outputFilePath = header + 'GridData/'
        else:
            filePath = header + '{}_data_before_time/'.format(args.model)
            outputFilePath = header + '{}GridData/'.format(args.model)
        filelist = findfile(filePath)
        print("Start preparing grid data...")
        for i in trange(0, len(filelist)):
            prepareGridData(filePath, filelist[i], outputFilePath, args.trajLen)
        print("Finish preparing grid data.")
    else:
        header = header2
        if args.SSN:
            for i in range(1,2):
                header_ = header + 'SSM_KNN/Database_{}/'.format(i+1)
                filePath = header_ + 'data_before_time/'
                outputFilePath = header_ + 'GridData/'
                filelist = findfile(filePath)
                print("Start preparing grid data...")
                for i in trange(0, len(filelist)):
                    prepareGridData(filePath, filelist[i], outputFilePath, args.trajLen)
                print("Finish preparing grid data.")
        else:
            filePath = header + '{}_data_before_time/'.format(args.model)
            outputFilePath = header + '{}GridData/'.format(args.model)
            filelist = findfile(filePath)
            print("Start preparing grid data...")
            for i in trange(0, len(filelist)):
                prepareGridData(filePath, filelist[i], outputFilePath, args.trajLen)
            print("Finish preparing grid data.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='train', choices=["train","experiment"], required=True)
    parser.add_argument('-s', '--SSN', type=bool, default=False, required=True)
    parser.add_argument('-l', '--trajLen', type=int, default=30, choices=[30,60], required=True)
    args = parser.parse_args()
    main(args)
    
            