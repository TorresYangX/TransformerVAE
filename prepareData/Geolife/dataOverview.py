import numpy as np
import pandas as pd

#Count all the data how many points per hour
# def overview(filePath):
#     number_of_node = [i for i in range(1)]
#     #The interval of the maximum difference between the latitude and longitude of the trajectory
#     diff_interval = pd.interval_range(start=0.001, end=0.805, freq=0.001, closed='left')
#     overview = pd.DataFrame(index=number_of_node, columns=['number', 'percent'])
#     trajectory_diff = pd.DataFrame(index=diff_interval, columns=['number', 'percent'])
#     for tmp1 in range(1):
#         overview.loc[tmp1].number = 0
#         overview.loc[tmp1].percent = 0
#     for tmp2 in diff_interval:
#         trajectory_diff.loc[tmp2].number = 0
#         trajectory_diff.loc[tmp2].percent = 0
#     for id_num in range(1, 10358):#id_num
#         print(id_num,'/10357')
#         file = str(id_num) + '.txt'
#         f=open(filePath+file,'r')
#         rawData=pd.read_csv(f,names=['id', 'timestep', 'lon', 'lat'])
#         rawData['timestep'] = pd.to_datetime(rawData.timestep)
#         for i in range(2, 9):
#             for j in range(0,24):
#                 timeStart = pd.to_datetime('2008-2-{} {}:00:00'.format(i, j))
#                 timeEnd = pd.to_datetime('2008-2-{} {}:59:59'.format(i, j))
#                 tempData = rawData[(rawData.timestep>=timeStart) & (rawData.timestep<=timeEnd)]
#     #             if tempData.lon.max() <= 116.8 and tempData.lon.min() >= 116.0 and tempData.lat.max() <= 40.2 and tempData.lat.min() >= 39.6:
#     #                #check if tempData.shape[0] is in the index of overview
#     #                 if tempData.shape[0] in number_of_node:
#     #                     overview.loc[tempData.shape[0]].number += 1
#     #                 else:
#     #                     #create new row
#     #                     overview.loc[tempData.shape[0]] = [1, 0]
#     #                     number_of_node.append(tempData.shape[0])
#     # total = overview.number.sum()
#     # for i in number_of_node:
#     #     overview.loc[i].percent = overview.loc[i].number/total
#     # # sort according to the percent
#     # overview = overview.sort_values(by='percent', ascending=False)
        
#     # #save the result
#     # overview.to_csv('../data/overview.csv', index=True)
#                 if tempData.lon.max() <= 116.8 and tempData.lon.min() >= 116.0 and tempData.lat.max() <= 40.2 and tempData.lat.min() >= 39.6 and tempData.shape[0]>=10:
#                     max_diff = max(tempData.lon.max()-tempData.lon.min(), tempData.lat.max()-tempData.lat.min())
#                     # judge which interval the max_diff belongs to
#                     for k in diff_interval:
#                         if max_diff in k:
#                             trajectory_diff.loc[k].number += 1
#                             break
#     total = trajectory_diff.number.sum()
#     for i in diff_interval:
#         trajectory_diff.loc[i].percent = trajectory_diff.loc[i].number/total
#     trajectory_diff.to_csv('../data/trajectory_diff.csv', index=True)

def modifyTraj(filePath):
    df = pd.read_csv(filePath, names=['number', 'percent'])
    df = df[1:]
    # change columns' type
    df['number'] = df['number'].astype('int')
    df['percent'] = df['percent'].astype('float')
    # create new column to store the sum of percent
    df['sum'] = 0
    df['sum'] = df['percent'].cumsum()
    # save the result
    df.to_csv(filePath, index=True)
    


if __name__ == '__main__':
    # filePath = '../data/seperatedData/'
    # overview(filePath)
    filePath = '../data/trajectory_diff.csv'
    modifyTraj(filePath)

