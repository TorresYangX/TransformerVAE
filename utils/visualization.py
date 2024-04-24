import math
import numpy as np
import matplotlib.pyplot as plt
from model_config import ModelConfig

db_size = [20,40,60,80,100]
ds_rate = [0.0,0.1,0.2]


def emd_chart(var_tp):
    NVAE_emd_folder = ModelConfig.NVAE.checkpoint_dir + '/emd'
    t2vec_emds_folder = ModelConfig.t2vec.checkpoint_dir + '/emd'
    NVAE_emds = []
    t2vec_emds = []
    for n_db in db_size:
        dataset_name = var_tp + str(n_db) + 'K'
        emd_NVAE = np.load(NVAE_emd_folder + '/emd_{}.npy'.format(dataset_name))
        emd_t2vec = np.load(t2vec_emds_folder + '/emd_{}.npy'.format(dataset_name))
        NVAE_emds.append(math.log(emd_NVAE))
        t2vec_emds.append(math.log(emd_t2vec))
    
    # 显示点并且连接到线用虚线
    plt.plot(db_size, NVAE_emds, 'ro-', label='NVAE')
    plt.plot(db_size, t2vec_emds, 'bo-', label='t2vec')
    
    plt.xlabel('database size(K)')
    plt.ylabel('EMD')
    plt.legend()
    plt.show()
    plt.savefig('emd_chart_db.png')

def NMD_chart(var_tp):
    NVAE_yao_folder = ModelConfig.NVAE.checkpoint_dir + '/yao'
    t2vec_yao_folder = ModelConfig.t2vec.checkpoint_dir + '/yao'
    NVAE_NMDs = []
    t2vec_NMDs = []
    # for n_db in db_size:
    #     dataset_name = var_tp + str(n_db) + 'K'
    #     NMD_NVAE = 0
    #     NMD_t2vec = 0
    #     with open(NVAE_yao_folder + '/MD_NMD_{}.csv'.format(dataset_name), 'r') as f:
    #         for line in f:
    #             if 'mass similarity(NMD)' in line:
    #                 NMD_NVAE = float(line.split(':')[1])
    #                 break
    #     with open(t2vec_yao_folder + '/MD_NMD_{}.csv'.format(dataset_name), 'r') as f:
    #         for line in f:
    #             if 'mass similarity(NMD)' in line:
    #                 NMD_t2vec = float(line.split(':')[1])
    #                 break
    #     NVAE_NMDs.append(NMD_NVAE)
    #     t2vec_NMDs.append(NMD_t2vec)
    
    # # 显示点并且连接到线用虚线, 点粗一点
    # plt.plot(db_size, NVAE_NMDs, 'ro-.', label='NVAE')
    # plt.plot(db_size, t2vec_NMDs, 'bo-.', label='t2vec')
    # plt.xlabel('database size(K)')
    # plt.ylabel('NMD')
    # plt.legend()
    # plt.show()
    # plt.savefig('NMD_chart_db.png')
    
    for v_ds in ds_rate:
        NMD_NVAE = 0
        NMD_t2vec = 0
        dataset_name = var_tp + str(v_ds)
        with open(NVAE_yao_folder + '/MD_NMD_{}.csv'.format(dataset_name), 'r') as f:
            for line in f:
                if 'mass similarity(NMD)' in line:
                    NMD_NVAE = float(line.split(':')[1])
                    break
        with open(t2vec_yao_folder + '/MD_NMD_{}.csv'.format(dataset_name), 'r') as f:
            for line in f:
                if 'mass similarity(NMD)' in line:
                    NMD_t2vec = float(line.split(':')[1])
                    break
        NVAE_NMDs.append(NMD_NVAE)
        t2vec_NMDs.append(NMD_t2vec)
        
    # 显示点并且连接到线用虚线, 点粗一点
    plt.plot(ds_rate, NVAE_NMDs, 'ro-.', label='NVAE')
    plt.plot(ds_rate, t2vec_NMDs, 'bo-.', label='t2vec')
    plt.xlabel('downsampling rate')
    plt.ylabel('NMD')
    plt.legend()
    plt.show()
    plt.savefig('NMD_chart_ds.png')
    
def NMA_chart(var_tp):
    NVAE_yao_folder = ModelConfig.NVAE.checkpoint_dir + '/yao'
    t2vec_yao_folder = ModelConfig.t2vec.checkpoint_dir + '/yao'
    NVAE_NMAs = []
    t2vec_NMAs = []
    # for n_db in db_size:
    #     dataset_name = var_tp + str(n_db) + 'K'
    #     NMA_NVAE = 0
    #     NMA_t2vec = 0
    #     with open(NVAE_yao_folder + '/MD_NMD_{}.csv'.format(dataset_name), 'r') as f:
    #         for line in f:
    #             if 'incluveness(NMA)' in line:
    #                 NMA_NVAE = float(line.split(':')[1])
    #                 break
    #     with open(t2vec_yao_folder + '/MD_NMD_{}.csv'.format(dataset_name), 'r') as f:
    #         for line in f:
    #             if 'incluveness(NMA)' in line:
    #                 NMA_t2vec = float(line.split(':')[1])
    #                 break
    #     NVAE_NMAs.append(NMA_NVAE)
    #     t2vec_NMAs.append(NMA_t2vec)
    
    # # 显示点并且连接到线用虚线, 点粗一点
    # plt.plot(db_size, NVAE_NMAs, 'ro-.', label='NVAE')
    # plt.plot(db_size, t2vec_NMAs, 'bo-.', label='t2vec')
    # plt.xlabel('database size(K)')
    # plt.ylabel('NMA')
    # plt.legend()
    # plt.show()
    # plt.savefig('NMA_chart_db.png')
    
    for v_ds in ds_rate:
        NMA_NVAE = 0
        NMA_t2vec = 0
        dataset_name = var_tp + str(v_ds)
        with open(NVAE_yao_folder + '/MD_NMD_{}.csv'.format(dataset_name), 'r') as f:
            for line in f:
                if 'incluveness(NMA)' in line:
                    NMA_NVAE = float(line.split(':')[1])
                    break
        with open(t2vec_yao_folder + '/MD_NMD_{}.csv'.format(dataset_name), 'r') as f:
            for line in f:
                if 'incluveness(NMA)' in line:
                    NMA_t2vec = float(line.split(':')[1])
                    break
        NVAE_NMAs.append(NMA_NVAE)
        t2vec_NMAs.append(NMA_t2vec)
        
    # 显示点并且连接到线用虚线, 点粗一点
    plt.plot(ds_rate, NVAE_NMAs, 'ro-.', label='NVAE')
    plt.plot(ds_rate, t2vec_NMAs, 'bo-.', label='t2vec')
    plt.xlabel('downsampling rate')
    plt.ylabel('NMA')
    plt.legend()
    plt.show()
    plt.savefig('NMA_chart_ds.png')
    
    
def RRNSA_chart(var_tp):
    NVAE_yao_folder = ModelConfig.NVAE.checkpoint_dir + '/yao'
    t2vec_yao_folder = ModelConfig.t2vec.checkpoint_dir + '/yao'
    NVAE_RRNSAs = []
    t2vec_RRNSAs = []
    # for n_db in db_size:
    #     dataset_name = var_tp + str(n_db) + 'K'
    #     RRNSA_NVAE = 0
    #     RRNSA_t2vec = 0
    #     with open(NVAE_yao_folder + '/MD_NMD_{}.csv'.format(dataset_name), 'r') as f:
    #         for line in f:
    #             if 'structure similarity(RRNSA)' in line:
    #                 RRNSA_NVAE = float(line.split(':')[1])
    #                 break
    #     with open(t2vec_yao_folder + '/MD_NMD_{}.csv'.format(dataset_name), 'r') as f:
    #         for line in f:
    #             if 'structure similarity(RRNSA)' in line:
    #                 RRNSA_t2vec = float(line.split(':')[1])
    #                 break
    #     NVAE_RRNSAs.append(RRNSA_NVAE)
    #     t2vec_RRNSAs.append(RRNSA_t2vec)
    
    # # 显示点并且连接到线用虚线, 点粗一点
    # plt.plot(db_size, NVAE_RRNSAs, 'ro-.', label='NVAE')
    # plt.plot(db_size, t2vec_RRNSAs, 'bo-.', label='t2vec')
    # plt.xlabel('database size(K)')
    # plt.ylabel('RRNSA')
    # plt.legend()
    # plt.show()
    # plt.savefig('RRNSA_chart_db.png')
    
    for v_ds in ds_rate:
        RRNSA_NVAE = 0
        RRNSA_t2vec = 0
        dataset_name = var_tp + str(v_ds)
        with open(NVAE_yao_folder + '/MD_NMD_{}.csv'.format(dataset_name), 'r') as f:
            for line in f:
                if 'structure similarity(RRNSA)' in line:
                    RRNSA_NVAE = float(line.split(':')[1])
                    break
        with open(t2vec_yao_folder + '/MD_NMD_{}.csv'.format(dataset_name), 'r') as f:
            for line in f:
                if 'structure similarity(RRNSA)' in line:
                    RRNSA_t2vec = float(line.split(':')[1])
                    break
        NVAE_RRNSAs.append(RRNSA_NVAE)
        t2vec_RRNSAs.append(RRNSA_t2vec)
        
    # 显示点并且连接到线用虚线, 点粗一点
    plt.plot(ds_rate, NVAE_RRNSAs, 'ro-.', label='NVAE')
    plt.plot(ds_rate, t2vec_RRNSAs, 'bo-.', label='t2vec')
    plt.xlabel('downsampling rate')
    plt.ylabel('RRNSA')
    plt.legend()
    plt.show()
    plt.savefig('RRNSA_chart_ds.png')
    
    
    
        