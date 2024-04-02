import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import pandas as pd
from config import Config
import matplotlib.pyplot as plt
from utils.pkl2csv import pkl2csv
from utils import preprocessing_proto
from utils import preprocessing_geolife
from utils.renumberTrajectories import renumberTrajectories


Config.dataset = 'geolife'
Config.post_value_updates()
Config.to_str()
# preprocessing_proto.clean_and_output_data()
# preprocessing_proto.generate_lonlat_data()
# preprocessing_proto.generate_grid_data()

# preprocessing_geolife.clean_and_output_data()
# preprocessing_geolife.generate_lonlat_data()
# preprocessing_geolife.generate_grid_data()


pkl2csv(Config.lonlat_ground_file, Config.lonlat_ground_file[:-4] + '.csv')
renumberTrajectories(Config.lonlat_ground_file[:-4] + '.csv', Config.lonlat_ground_file[:-4] + '_renumbered.csv')


