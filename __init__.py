from utils import preprocessing_proto
from config import Config
import pickle


Config.dataset = 'porto'
Config.post_value_updates()
Config.to_str()
# preprocessing_proto.clean_and_output_data()
# preprocessing_proto.generate_lonlat_data()
data = pickle.load(open(Config.dataset_file, 'rb'))
print(data.head())
print(len(data['wgs_seq'][0]))
