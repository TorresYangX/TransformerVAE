import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import pandas as pd
from dataset_config import DatasetConfig
from utils import preprocessing_porto
from utils import preprocessing_geolife
from utils.seq2multi_row import seq2multi_row


DatasetConfig.dataset = 'porto'
DatasetConfig.post_value_updates()
# preprocessing_porto.clean_and_output_data()
# preprocessing_porto.generate_intepolation_data()
# preprocessing_porto.generate_lonlat_data()
# preprocessing_porto.generate_grid_data()

# preprocessing_geolife.clean_and_output_data()
# preprocessing_geolife.generate_intepolation_data()
# preprocessing_geolife.generate_lonlat_data()
# preprocessing_geolife.generate_grid_data()

# seq2multi_row(DatasetConfig.lonlat_ground_file, DatasetConfig.lonlat_ground_file.replace('.pkl', '_multirow.csv'))

# data = pd.read_pickle(DatasetConfig.lonlat_ground_file)
# print(data.head())

# seq2multi_row('exp\porto\\t2vec\emd\\retrieve_trajs.csv', 'exp\porto\\t2vec\emd\\retrieve_trajs_multirow.csv')

train_generator = preprocessing_porto.train_dataset_generator()
train_generator.generate()