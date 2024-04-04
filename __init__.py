import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import pandas as pd
from dataset_config import DatasetConfig
import matplotlib.pyplot as plt
from utils.pkl2csv import pkl2csv
from utils import preprocessing_porto
from utils import preprocessing_geolife
from utils.seq2multi_row import seq2multi_row


DatasetConfig.dataset = 'geolife'
DatasetConfig.post_value_updates()
# preprocessing_porto.clean_and_output_data()
# preprocessing_porto.generate_intepolation_data()
# preprocessing_porto.generate_lonlat_data()
# preprocessing_porto.generate_grid_data()

# preprocessing_geolife.clean_and_output_data()
# preprocessing_geolife.generate_intepolation_data()
# preprocessing_geolife.generate_lonlat_data()
# preprocessing_geolife.generate_grid_data()

# seq2multi_row(Config.lonlat_ground_file, Config.lonlat_ground_file.replace('.pkl', '_multirow.csv'))

# data = pd.read_pickle(Config.grid_ground_file)
# print(data.head())