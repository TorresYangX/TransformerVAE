import os
import time
import logging
import argparse
import numpy as np
import pandas as pd
from pyemd import emd
from model_config import ModelConfig
from dataset_config import DatasetConfig

logging.getLogger().setLevel(logging.INFO)

def cityEMD(ground_trajs, reterieved_trajs, emd_folder, dataset_name, NLAT=50, NLON=50):
    
    def normalize(trajectories):
        wgs_seq = np.array(trajectories['wgs_seq'].tolist())
        wgs_seq[:,:,0] = (wgs_seq[:,:,0] - DatasetConfig.min_lon) / (DatasetConfig.max_lon - DatasetConfig.min_lon) 
        wgs_seq[:,:,1] = (wgs_seq[:,:,1] - DatasetConfig.min_lat) / (DatasetConfig.max_lat - DatasetConfig.min_lat)
        
        return wgs_seq
    
    def compute_emd(flowReal_c, flowRetrieved_c):
        emd_value = sum(emd(np.ascontiguousarray(flowReal_c[it, :]), 
                        np.ascontiguousarray(flowRetrieved_c[it, :]), 
                       flowDistance) for it in range(NLAT*NLON))
        return emd_value
    
    
    s_time = time.time()
    logging.info('[Data Normalize] start')
    targetX = normalize(ground_trajs)
    retrievedY = normalize(reterieved_trajs)
    Xdis = (targetX[:, :, 0]*NLON + targetX[:, :, 1]*(NLAT-1)*NLON).astype(int)
    Ydis = (retrievedY[:, :, 0]*NLON + retrievedY[:, :, 1]*(NLAT-1)*NLON).astype(int)
    n_time = time.time()
    
    logging.info('[Flow Matrix] start')
    flowReal = np.zeros((NLAT*NLON,NLAT*NLON))
    flowRetrieved = np.zeros((NLAT*NLON,NLAT*NLON))
    flowDistance = np.zeros((NLAT*NLON,NLAT*NLON))
    latitudes = np.arange(NLAT*NLON) // NLON
    longitudes = np.arange(NLAT*NLON) % NLON
    flowDistance = np.minimum(10.0, np.sqrt((latitudes[:, None] - latitudes)**2 + (longitudes[:, None] - longitudes)**2))
    for k in range(ModelConfig.NVAE.traj_len-1):
        flowReal[Xdis[:,k],Xdis[:,k+1]] += 1.0
        flowRetrieved[Ydis[:,k],Ydis[:,k+1]] += 1.0
    f_time = time.time()
        
    logging.info('[EMD Compute] start')
    emd_ = compute_emd(flowReal, flowRetrieved)
    logging.info('EMD: {}'.format(emd_))
    np.save(emd_folder+'/emd_{}.npy'.format(dataset_name), emd_)

    logging.info('normalize time: {:.2f}s, flow time: {:.2f}s, emd time: {:.2f}s'.format(n_time-s_time, f_time-n_time, time.time()-f_time))
    return


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
    emd_folder = config_class.checkpoint_dir + '/emd'
    os.makedirs(emd_folder, exist_ok=True)
    
    db_size = [20,40,60,80]
    ds_rate = []
    dt_rate = []
    
    for n_db in db_size:
        dataset_name = 'db_{}K'.format(n_db)
        logging.info('%s start' % dataset_name)
        retrieve_trajs = pd.read_pickle(retrieve_folder + '/retr_trajs_{}.pkl'.format(dataset_name))
        ground_trajs = pd.read_pickle(DatasetConfig.dataset_folder 
                                      +'/{}/lonlat/{}_ground.pkl'.format(dataset_name, DatasetConfig.dataset_prefix))
        ground_trajs = ground_trajs.iloc[:retrieve_trajs.shape[0]]
        cityEMD(ground_trajs, retrieve_trajs, emd_folder, dataset_name)
    