import pandas as pd
import numpy as np
import os
from tqdm import trange
import argparse


# conbine all the trajectories in the same day
def getGroundTruth(args, groundTruthPath, dataPath):
    length = 0
    day = args.day
    if not os.path.exists(groundTruthPath):
        trajectories = pd.DataFrame()
        for i in trange(24):
            file = '{}_{}.csv'.format(day, i)
            if os.path.exists(dataPath + file):
                temp = pd.read_csv(dataPath + file, header = None)
                length += len(temp)
                trajectories = pd.concat([trajectories, temp], axis=0)
        trajectories = trajectories.reset_index(drop=True)
        with open(groundTruthPath, mode='w') as f:
            trajectories.to_csv(f, header = None, index = False)
            print('length of ground truth: {}'.format(length))
    else:
        trajectories = pd.read_csv(groundTruthPath, header = None)
    return trajectories

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--day', type=str, default='2', choices=["2","3","4","5","6","7","8"], required=True, help="day of the data")
    args = parser.parse_args()
    groundTruthPath = '../data/Experiment/groundTruth'
    if not os.path.exists(groundTruthPath):
        os.mkdir(groundTruthPath)
    groundTruthPath = '../data/Experiment/groundTruth/groundTruth_{}.csv'.format(args.day)
    dataPath = '../data/Experiment/history/history_data_before_time/'
    getGroundTruth(args, groundTruthPath, dataPath)
