import os
import logging
import argparse
import pandas as pd
from sklearn.neighbors import KDTree
from model_config import ModelConfig
from dataset_config import DatasetConfig

logging.getLogger().setLevel(logging.INFO)


def data_index_load(model_config, dataset_name, total_data_path, test_data_path):
    total_data = pd.read_pickle(total_data_path)
    total_data = total_data.iloc[:total_data.shape[0]-(total_data.shape[0] % model_config.BATCH_SIZE)]
    total_data = total_data.reset_index(drop=True)
    test_data = pd.read_pickle(test_data_path)
    test_data = test_data.iloc[:test_data.shape[0]-(test_data.shape[0] % model_config.BATCH_SIZE)]
    test_data = test_data.reset_index(drop=True)
    
    total_index_dict = {}
    test_index_dict = {} 
    total_index_root_folder = model_config.index_dir + '/train/' # for ds and dt, db is different
    test_index_root_folder = model_config.index_dir + '/{}/'.format(dataset_name)
    for dirs in os.listdir(total_index_root_folder):
        total_index_dict[dirs] = pd.read_csv(total_index_root_folder + dirs + '/total_index.csv', header=None)
    for dirs in os.listdir(test_index_root_folder):
        test_index_dict[dirs] = pd.read_csv(test_index_root_folder + dirs + '/test_index.csv', header=None)
    return total_data, test_data, total_index_dict, test_index_dict

def retrieve(total_indexs, test_indexs, retr_num, retr_file, total_data):
    logging.info('Begin to retrieve...')
    tree = KDTree(total_indexs)
    solution = []
    for i in range(len(test_indexs)):
        _, nearest_ind = tree.query(test_indexs[i].reshape(1,test_indexs.shape[1]), k=retr_num)
        solution += list(nearest_ind[0])
    logging.info('End to retrieve, begin to save...')
    save_retr_traj(retr_file, total_data, solution)
    return 0

def save_retr_traj(retr_file, total_data, solution):
    retr_trajs= [pd.DataFrame(total_data.iloc[i]).T for i in solution]
    combined_data = pd.concat(retr_trajs, axis=0, ignore_index=True)
    combined_data.to_pickle(retr_file)
    return 0
    
    
def pipline(dataset_name):
    logging.info('[%s retrieve] Start' % dataset_name)
    
    total_data = DatasetConfig.dataset_folder + '/train/lonlat/{}_total.pkl'.format(DatasetConfig.dataset_prefix) # for ds and dt, db is different
    ground_data = DatasetConfig.dataset_folder + '/train/lonlat/{}_ground.pkl'.format(DatasetConfig.dataset_prefix) # for ds and dt, db is different
    test_data = DatasetConfig.dataset_folder + '/{}/lonlat/{}_test.pkl'.format(dataset_name, DatasetConfig.dataset_prefix)
    
    total_data, test_data, total_index_dict, test_index_dict = data_index_load(config_class, dataset_name, total_data, test_data)
    logging.info('total_data shape: %s' % str(total_data.shape))
    logging.info('test_data shape: %s' % str(test_data.shape))
    total_indexs = pd.concat(total_index_dict.values(), axis=1).values
    test_indexs = pd.concat(test_index_dict.values(), axis=1).values
    
    retrieve_trajs_file = retrieve_folder + '/retr_trajs_{}.pkl'.format(dataset_name)
    
    ground_data_df = pd.read_pickle(ground_data)
    ground_data_len = ground_data_df.shape[0] - ground_data_df.shape[0] % test_data.shape[0]
    logging.info('ground_data_len: %d' % ground_data_len)
    
    retr_num = int(ground_data_len / test_data.shape[0])
    logging.info('retr_num: %d' % retr_num)
    
    retrieve(total_indexs, test_indexs, retr_num, retrieve_trajs_file, total_data)
    

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
    
    retrieve_folder = config_class.checkpoint_dir + '/retrieve'
    
    os.makedirs(retrieve_folder, exist_ok=True)
    
    db_size = [20,40,60,80,100]
    ds_rate = [0.1,0.2,0.3,0.4,0.5]
    dt_rate = []
    
    # for n_db in db_size:
    #     dataset_name = 'db_{}K'.format(n_db)
    #     pipline(dataset_name)
    
    for v_ds in ds_rate:
        dataset_name = 'ds_{}'.format(v_ds)
        pipline(dataset_name)
        
    # for v_dt in dt_rate:
    #     dataset_name = 'dt_{}'.format(v_dt)
    #     pipline(dataset_name)