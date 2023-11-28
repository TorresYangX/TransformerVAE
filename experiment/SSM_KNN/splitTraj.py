import pandas as pd
import numpy as np
import os
from tqdm import trange
import argparse

def splitExperimentTraj(orginalDataPath, DataBase_1_path, DataBase_2_path):
    for day in ['2','3','4','5','6','7','8']:
        for i in trange(24):
            file = '{}_{}.csv'.format(day, i)
            if os.path.exists(orginalDataPath + file):
                temp = pd.read_csv(orginalDataPath + file, header = None)
                temp = temp.reset_index(drop=True)
                line_num_1=[]
                line_num_2=[]
                for j in range(0,len(temp),2):
                    line_num_1.append(j)
                for j in range(1,len(temp),2):
                    line_num_2.append(j)
                with open(DataBase_1_path + '{}_{}.csv'.format(day, i), mode='a') as f:
                    temp.iloc[line_num_1].to_csv(f, header = None, index = False)
            
                with open(DataBase_2_path + '{}_{}.csv'.format(day, i), mode='a') as f:
                    temp.iloc[line_num_2].to_csv(f, header = None, index = False)

def splitTrainTraj(orginalDataPath, DataBase_path):
    for day in ['2','3','4','5','6','7','8']:
        for i in trange(24):
            file = '{}_{}.csv'.format(day, i)
            if os.path.exists(orginalDataPath + file):
                temp = pd.read_csv(orginalDataPath + file, header = None)
                temp = temp.reset_index(drop=True)
                line_num=[]
                for j in range(0,len(temp),2):
                    line_num.append(j)
                with open(DataBase_path + '{}_{}.csv'.format(day, i), mode='a') as f:
                    temp.iloc[line_num].to_csv(f, header = None, index = False)

def main(model):
    path = '../data/{}/SSM_KNN/'.format(model)
    if not os.path.exists(path):
        os.mkdir(path)
    if model == 'Train':
        DataBase_path = '../data/{}/SSM_KNN/DataBase/'.format(model)
        if not os.path.exists(DataBase_path):
            os.mkdir(DataBase_path)
        DataBase_path = '../data/{}/SSM_KNN/DataBase/data_before_time/'.format(model)
        if not os.path.exists(DataBase_path):
            os.mkdir(DataBase_path)
        orginalDataPath = '../data/{}/{}_data_before_time/'.format(model, model.lower())
        splitTrainTraj(orginalDataPath, DataBase_path)

    else:
        DataBase_1_path = '../data/{}/SSM_KNN/DataBase_1/'.format(model)
        DataBase_2_path = '../data/{}/SSM_KNN/DataBase_2/'.format(model)
        if not os.path.exists(DataBase_1_path):
            os.mkdir(DataBase_1_path)
        if not os.path.exists(DataBase_2_path):
            os.mkdir(DataBase_2_path)
        DataBase_1_path = '../data/{}/SSM_KNN/DataBase_1/data_before_time/'.format(model)
        DataBase_2_path = '../data/{}/SSM_KNN/DataBase_2/data_before_time/'.format(model)
        if not os.path.exists(DataBase_1_path):
            os.mkdir(DataBase_1_path)
        if not os.path.exists(DataBase_2_path):
            os.mkdir(DataBase_2_path)
        orginalDataPath = '../data/{}/{}_data_before_time/'.format(model, model.lower())
        splitExperimentTraj(orginalDataPath, DataBase_1_path, DataBase_2_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, default='Train', choices=['Train', 'Experiment'], required=True, help="train or experiment")

    args = parser.parse_args()

    main(args.model)

    
