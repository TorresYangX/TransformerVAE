import numpy as np
import pandas as pd
import os
    

if __name__ == '__main__':
    folder1 = '../data/Geolife/Train/train_data_before_time/'
    folder2 = '../data/Geolife/Experiment/experiment_data_before_time/'
    
    # read all the .csv files in the folder, and count the number of rows in each file
    totalRows = 0
    for folder in [folder1, folder2]:
        print('Folder:', folder)
        for file in os.listdir(folder):
            filename = folder + file
            df = pd.read_csv(filename,header=None, names=['lat', 'lon', 'time', 'label'])
            print(file, len(df))
            totalRows += len(df)
            
    print('Total rows:', totalRows)

