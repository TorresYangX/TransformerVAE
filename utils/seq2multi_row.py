import pandas as pd
from tqdm import tqdm
from model_config import ModelConfig

ID = -1

def seq2multi_row(file_path, OutputFile):
    
    def convert_multi_row(row):
        global ID
        ID += 1
        return [[ID, coord[0], coord[1], pd.to_datetime(row['timestamp'])] for coord in row['wgs_seq']]
    
    
    df = pd.read_pickle(file_path)
    totalID = df.shape[0]
    df = df[['TAXI_ID', 'wgs_seq', 'timestamp']]
    df = df.apply(convert_multi_row, axis=1).explode().tolist()
    df = pd.DataFrame(df, columns=['TAXI_ID', 'lon', 'lat', 'timestamp'])
    # write out data
    with open(OutputFile, mode='w') as f:
        df.to_csv(OutputFile, header=None, index=False)
    
    