import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import pandas as pd
from dataset_config import DatasetConfig
from model_config import ModelConfig
from utils import preprocessing_porto
from utils import preprocessing_geolife
from utils.seq2multi_row import seq2multi_row
from utils import visualization


DatasetConfig.dataset = 'porto'
DatasetConfig.post_value_updates()

# train_generator = preprocessing_porto.downsampling_dataset_generator(0.5)
# train_generator.generate()

data = pd.read_pickle(DatasetConfig.dataset_folder+'/db_20K/lonlat/porto_test.pkl')
print(data.head())
# seq2multi_row(DatasetConfig.dataset_folder+'/db_20K/porto.pkl', DatasetConfig.dataset_folder+'/db_20K/porto.csv')

# visualization.emd_chart('db_')
# visualization.yao_chart('NMD', 'ds_')

