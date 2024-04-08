import os
import logging
import argparse
import pandas as pd
from sklearn.neighbors import KDTree
from model_config import ModelConfig
from dataset_config import DatasetConfig

logging.getLogger().setLevel(logging.INFO)


def data_index_load(model_config, total_data_path, test_data_path):
    total_data = pd.read_pickle(total_data_path)
    total_data = total_data.iloc[:-(total_data.shape[0] % model_config.BATCH_SIZE)]
    total_data = total_data.reset_index(drop=True)
    test_data = pd.read_pickle(test_data_path)
    test_data = test_data.iloc[:-(test_data.shape[0] % model_config.BATCH_SIZE)]
    test_data = test_data.reset_index(drop=True)
    
    total_index_dict = {}
    test_index_dict = {}
    for dirs in os.listdir(model_config.index_dir):
        total_index_dict[dirs] = pd.read_csv(model_config.index_dir + '/' + dirs + '/total_index.csv')
        test_index_dict[dirs]=pd.read_csv(model_config.index_dir + '/' + dirs + '/test_index.csv', header=None)
    return total_data, test_data, total_index_dict, test_index_dict

def retrieve(total_indexs, test_indexs, retr_num, retr_file, total_data):
    logging.info('Begin to retrieve...')
    tree = KDTree(total_indexs)
    solution = []
    for i in range(len(test_indexs)):
        nearest_dist, nearest_ind = tree.query(test_indexs[i].reshape(1,test_indexs.shape[1]), k=retr_num)
        meanLoss = nearest_dist[0].mean()
        solution += list(nearest_ind[0])
    logging.info('End to retrieve, begin to save...')
    save_retr_traj(retr_file, total_data, solution)
    return 0

def save_retr_traj(retr_file, total_data, solution):
    retr_trajs= [pd.DataFrame(total_data.iloc[i]).T for i in solution]
    combined_data = pd.concat(retr_trajs, axis=0, ignore_index=True)
    with open(retr_file, mode='w') as f:
        combined_data.to_csv(f, header=None, index=False)
    logging.info('Save retrieve trajs to %s' % retr_file)
    return 0
    
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='')
    parser.add_argument("--dataset", type=str, help='')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    DatasetConfig.update(dict(filter(lambda kv: kv[1] is not None, vars(args).items())))
    
    model_mapping = {
        'NVAE': {'config': ModelConfig.NVAE},
        'AE': {'config': ModelConfig.AE},
        'VAE': {'config': ModelConfig.VAE},
        'Transformer': {'config': ModelConfig.Transformer},
        't2vec': {'config': ModelConfig.t2vec}
    }
    if args.model not in model_mapping:
        raise ValueError('model not found')
    config_class = model_mapping[args.model]['config']
    
    emd_folder = config_class.checkpoint_dir + '/emd'
    if not os.path.exists(emd_folder):
        os.mkdir(emd_folder)
    
    total_data = DatasetConfig.lonlat_total_file
    ground_data = DatasetConfig.lonlat_ground_file
    test_data = DatasetConfig.lonlat_test_file
    
    total_data, test_data, total_index_dict, test_index_dict = data_index_load(config_class, total_data, test_data)
    total_indexs = pd.concat(total_index_dict.values(), axis=1).values
    test_indexs = pd.concat(test_index_dict.values(), axis=1).values
    
    retrieve_trajs_file = emd_folder + '/retrieve_trajs.pkl'
    
    ground_data_df = pd.read_pickle(ground_data)
    ground_data_len = ground_data_df.shape[0] - ground_data_df.shape[0] % test_data.shape[0]
    logging.info('ground_data_len: %d' % ground_data_len)
    
    retr_num = int(ground_data_len / test_data.shape[0])
    logging.info('retr_num: %d' % retr_num)
    
    retrieve(total_indexs, test_indexs, retr_num, retrieve_trajs_file, total_data)
    retrievalLen = len(pd.read_csv(retrieve_trajs_file, header = None))
    logging.info('retrievalLen: %d' % retrievalLen)