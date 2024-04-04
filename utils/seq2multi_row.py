import pandas as pd
from model_config import ModelConfig


def seq2multi_row(file_path, OutputFile):
    def convert_multi_row(row):
        return [[row['TAXI_ID'], coord[0], coord[1], pd.to_datetime(row['timestamp'])] for coord in row['wgs_seq']]
    
    
    df = pd.read_pickle(file_path)
    df.columns = ['TAXI_ID', 'wgs_seq', 'timestamp']
    df = df[['TAXI_ID', 'wgs_seq', 'timestamp']]
    df = df.apply(convert_multi_row, axis=1).explode().tolist()
    df = pd.DataFrame(df, columns=['TAXI_ID', 'lon', 'lat', 'timestamp'])
    totalID = int(df.shape[0]/ModelConfig.NVAE.traj_len)
    for i in range(totalID):
        for j in range(ModelConfig.NVAE.traj_len):
            df.loc[i*ModelConfig.NVAE.traj_len+j, 'TAXI_ID'] = i
    # write out data
    with open(OutputFile, mode='w') as f:
        df.to_csv(OutputFile, header=None, index=False)
    
    