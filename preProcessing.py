import os
import numpy as np
import pandas as pd
from tqdm import trange


# longtitude: 116 116.8
# latitude: 39.6 40.2
# prigin trajectory min length: 10
# min lon_lat diff = 0.04
# time interval: 1min
# time range: 2.1-2.8
# trajectory length: 60

train_start_id = 1
train_end_id = 9001
query_start_id = 9001
query_end_id = 10358

def preprocess(START, END, filePath, outputFilePath, startid, endid):
    for id_num in trange(startid, endid):
        file = str(id_num) + '.txt'
        f=open(filePath+file,'r')
        rawData=pd.read_csv(f,names=['id', 'timestep', 'lon', 'lat'])
        rawData['timestep'] = pd.to_datetime(rawData.timestep)
        for i in range(START, END):
            for j in range(0, 24):
                timeStart = pd.to_datetime('2008-2-{} {}:00:00'.format(i, j))
                timeEnd = pd.to_datetime('2008-2-{} {}:59:59'.format(i, j))
                tempData = rawData[(rawData.timestep>=timeStart) & (rawData.timestep<=timeEnd)]
                # check if longitude and latitude are in the range
                if tempData.lon.max() <= 116.8 and tempData.lon.min() >= 116.0 and tempData.lat.max() <= 40.2 and tempData.lat.min() >= 39.6 and tempData.shape[0] >= 10 and ((tempData.lon.max()-tempData.lon.min())>=0.04 or (tempData.lat.max()-tempData.lat.min())>=0.04):
                    id_set = tempData.id.drop_duplicates()
                    TIMEINTERVAL = 1
                    tt = range(0, 60, TIMEINTERVAL)
                    timeRange = pd.date_range(start=timeStart, periods=len(tt), freq='1min')
                    outputPath = outputFilePath + str(id_num) + '/'
                    fileName = str(i) + '_' + str(j) + '.csv'
                    if not os.path.exists(outputPath):
                        os.makedirs(outputPath)
                    for k in id_set:
                        partData = tempData[tempData.id==k]
                        partData = partData.reset_index()
                        lon = np.interp(tt, partData.timestep.dt.minute, partData.lon)
                        lat = np.interp(tt, partData.timestep.dt.minute, partData.lat)
                        res = pd.DataFrame([timeRange, lon, lat, [k]*int(len(tt))]).T
                        if True in res.isnull().values:
                            continue
                        else:
                            with open(outputPath + fileName, mode='a') as f:
                                res.to_csv(f, header = None, index = False)


def preprocess_time(START, END, filePath, outputFilePath, startid, endid):
    for i in range(START, END):
        for j in range(24):
            data = pd.DataFrame()
            for id_num in trange(startid, endid):
                filePath_ = filePath + str(id_num) + '/'
                singleFile = str(i) + '_' + str(j) + '.csv'
                if os.path.exists(filePath_+singleFile):
                    tempData = pd.read_csv(filePath_+singleFile, header = None)
                    df_check=tempData.isnull()
                    if True in df_check:
                        print(id_num, " has nan.")
                    else:
                        data = pd.concat([data,tempData])
                    
            fileName = str(i) + '_' + str(j) + '.csv'
            if data.shape[0] != 0:
                with open(outputFilePath + fileName, mode='a') as f:
                    data.to_csv(f, header = None, index = False)
                    

if __name__ == '__main__':
    START, END = 2, 9
    # filePath = '../data/seperatedData/'
    # outputFilePath = '../data/train_data_before/'
    # if not os.path.exists(outputFilePath):  
    #     os.makedirs(outputFilePath)
    # print('preProcessing traindata...')
    # preprocess(START, END, filePath, outputFilePath, train_start_id, train_end_id)
    # print('preProcessing traindata Done!')
    filePath = '../data/train_data_before/'
    outputFilePath = '../data/train_data_before_time/'
    if not os.path.exists(outputFilePath):
        os.makedirs(outputFilePath)
    print('preProcessing according time...')
    preprocess_time(START, END, filePath, outputFilePath, train_start_id, train_end_id)
    print('preProcessing according time Done!')

    # filePath = '../data/seperatedData/'
    # outputFilePath = '../data/query_data_before/'
    # if not os.path.exists(outputFilePath):  
    #     os.makedirs(outputFilePath)
    # print('preProcessing querydata...')
    # preprocess(START, END, filePath, outputFilePath, query_start_id, query_end_id)
    # print('preProcessing querydata Done!')
    filePath = '../data/query_data_before/'
    outputFilePath = '../data/query_data_before_time/'
    if not os.path.exists(outputFilePath):
        os.makedirs(outputFilePath)
    print('preProcessing according time...')
    preprocess_time(START, END, filePath, outputFilePath, query_start_id, query_end_id)
    print('preProcessing according time Done!')
    
