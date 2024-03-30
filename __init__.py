import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import pandas as pd
from config import Config
import matplotlib.pyplot as plt
from utils.pkl2csv import pkl2csv
from utils import preprocessing_proto
from utils import preprocessing_beijing
from utils.renumberTrajectories import renumberTrajectories


Config.dataset = 'beijing'
Config.post_value_updates()
Config.to_str()
# preprocessing_proto.clean_and_output_data()
# preprocessing_proto.generate_lonlat_data()
# preprocessing_proto.generate_grid_data()

# preprocessing_beijing.prepare_csv()
# preprocessing_beijing.clean_and_output_data()
# preprocessing_beijing.generate_lonlat_data()

pkl2csv(Config.lonlat_test_file, Config.lonlat_test_file.replace('.pkl', '.csv'))
renumberTrajectories(Config.lonlat_test_file.replace('.pkl', '.csv'), Config.lonlat_test_file.replace('.pkl', '.csv'))