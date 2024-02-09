import pandas as pd
import os

path = '../data/Porto/rawTimeData/'

totalRows = 0

for file in os.listdir(path):
    filename = path + file
    df = pd.read_csv(filename,header=None, names=['lat', 'lon', 'time'])
    print(file, len(df))
    totalRows += len(df)
    
print('Total rows:', totalRows)


