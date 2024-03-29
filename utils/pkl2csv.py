import pandas as pd

def pkl2csv(pkl_file, csv_file):
    df = pd.read_pickle(pkl_file)
    df.to_csv(csv_file, index=False, header=None)
    