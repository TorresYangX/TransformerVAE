import math
import numpy as np
import matplotlib.pyplot as plt
from model_config import ModelConfig
from dataset_config import DatasetConfig

db_size = [20,40,60,80,100]
ds_rate = [0.1,0.2,0.3,0.4,0.5]


def plot(x_list, y_dict, x_label, y_label, title):
    plt.figure()
    for key in y_dict.keys():
        plt.plot(x_list, y_dict[key], 'o-.', label=key)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(ModelConfig.NVAE.checkpoint_dir[:-4]+'visualization/' + title + '.png')
    plt.show()

def emd_chart(var_tp):
    NVAE_emd_folder = ModelConfig.NVAE.checkpoint_dir + '/emd'
    t2vec_emds_folder = ModelConfig.t2vec.checkpoint_dir + '/emd'
    NVAE_emds = []
    t2vec_emds = []
    type_dict = {
        'db_': (db_size, 'K', 'database size(K)'),
        'ds_': (ds_rate, '', 'downsampling rate')
    }
    for n_db in type_dict[var_tp][0]:
        dataset_name = var_tp + str(n_db) + type_dict[var_tp][1]
        emd_NVAE = np.load(NVAE_emd_folder + '/emd_{}.npy'.format(dataset_name))
        emd_t2vec = np.load(t2vec_emds_folder + '/emd_{}.npy'.format(dataset_name))
        NVAE_emds.append(math.log(emd_NVAE))
        t2vec_emds.append(math.log(emd_t2vec))
    plot_dict = {'NVAE': NVAE_emds, 't2vec': t2vec_emds}
    plot(type_dict[var_tp][0], plot_dict, type_dict[var_tp][2], 'log(EMD)', 'emd_chart_' + var_tp)
    
    
    
    
def yao_chart(metric_name, var_tp):
    NVAE_yao_folder = ModelConfig.NVAE.checkpoint_dir + '/yao'
    t2vec_yao_folder = ModelConfig.t2vec.checkpoint_dir + '/yao'
    NVAE_ = []
    t2vec_ = []
    type_dict = {
        'db_': (db_size, 'K', 'database size(K)'),
        'ds_': (ds_rate, '', 'downsampling rate')
    }
    metric_dict = {
        'NMD': 'mass similarity(NMD)',
        'NMA': 'incluveness(NMA)',
        'RRNSA': 'structure similarity(RRNSA)'
    }
    
    for n_db in type_dict[var_tp][0]:
        dataset_name = var_tp + str(n_db) + type_dict[var_tp][1]
        with open(NVAE_yao_folder + '/MD_NMD_{}.csv'.format(dataset_name), 'r') as f:
            for line in f:
                if metric_dict[metric_name] in line:
                    value = float(line.split(':')[1])
                    NVAE_.append(value)
                    break
        with open(t2vec_yao_folder + '/MD_NMD_{}.csv'.format(dataset_name), 'r') as f:
            for line in f:
                if metric_dict[metric_name] in line:
                    value = float(line.split(':')[1])
                    t2vec_.append(value)
                    break
    plot_dict = {'NVAE': NVAE_, 't2vec': t2vec_}
    plot(type_dict[var_tp][0], plot_dict, type_dict[var_tp][2], metric_name, metric_name + '_chart_' + var_tp)
     
    
    
    
        