import pandas as pd

# dataPath = '../data/Porto/Origin.csv'
# data = pd.read_csv(dataPath)
# # only need the row that missing_data column is False
# data = data[data['MISSING_DATA'] == False]
# data = data[['TAXI_ID', 'POLYLINE', 'TIMESTAMP']]

# # convert the unix time to datetime
# data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], unit='s')
# data = data[(data['TIMESTAMP'] >= '2013-07-01 00:00:00') & (data['TIMESTAMP'] <= '2013-07-31 23:59:59')]

# # save the data
# data.to_csv('../data/Porto/Train.csv', index=False)

dataPath = '../data/Porto/Train.csv'
data = pd.read_csv(dataPath)

# treat the POLYLINE column as list
data['POLYLINE'] = data['POLYLINE'].apply(lambda x: eval(x))
# only need row that POLYLINE length larger than 60, the POLYLINE format is [[-8.628561,41.158998],[-8.628229,41.159032],...]
data = data[data['POLYLINE'].apply(lambda x: len(x) > 60)]
print(data.shape)

