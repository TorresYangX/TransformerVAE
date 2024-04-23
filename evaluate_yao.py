import time
import math
import logging
import argparse
import numpy as np
import pandas as pd
from model_config import ModelConfig
from dataset_config import DatasetConfig

logging.getLogger().setLevel(logging.INFO)

def get_OD_matrix(file_path):
    wgs_seq_df = pd.read_pickle(file_path)['wgs_seq']
    wgs_seq = np.array(wgs_seq_df.tolist())
    OD_grid = np.zeros((DatasetConfig.grid_num, DatasetConfig.grid_num, DatasetConfig.grid_num, DatasetConfig.grid_num))
    wgs_seq[:,:,0] = np.floor((wgs_seq[:,:,0] - DatasetConfig.min_lon) / DatasetConfig.grid_size)
    wgs_seq[:,:,1] = np.floor((wgs_seq[:,:,1] - DatasetConfig.min_lat) / DatasetConfig.grid_size)
    __data = np.zeros((wgs_seq.shape[0], wgs_seq.shape[1]-1, 2, 2))
    for i in range(wgs_seq.shape[1]-1):
        __data[:, i, 0, 0] = wgs_seq[:, i, 0]
        __data[:, i, 0, 1] = wgs_seq[:, i, 1]
        __data[:, i, 1, 0] = wgs_seq[:, i+1, 0]
        __data[:, i, 1, 1] = wgs_seq[:, i+1, 1]
        
    i_indices = __data[:, :, 0, 0].astype(int)
    j_indices = __data[:, :, 0, 1].astype(int)
    k_indices = __data[:, :, 1, 0].astype(int)
    l_indices = __data[:, :, 1, 1].astype(int)

    np.add.at(OD_grid, (i_indices, j_indices, k_indices, l_indices), 1)
    return OD_grid


def find_path(i,n):
    global adj_matrix
    global number_left
    global number_right
    global label_left
    global label_right
    global match_right
    global visit_left
    global visit_right
    global slack_right
    global last_v_l
    global last_v_r
    
    if n == 0:
        return 0
    for j, match_weight in enumerate(adj_matrix[i]):
        if visit_right[j]: continue  
        gap = label_left[i] + label_right[j] - match_weight
        if gap == 0:
            if number_right[j] != 0:  
                free_n = min(number_right[j],n)
                number_right[j] -= free_n
                if number_right[j] == 0:
                    visit_right[j] = True
                n -= free_n
                if i in match_right[j]:
                    match_right[j][i] += free_n
                else:
                    match_right[j][i] = free_n
                if n == 0:
                    return 0
            for k in list(match_right[j].keys()):
                visit_right[j] = True
                if k == i:
                    continue
                free_n = min(match_right[j][k],n)
                while True:
                    find_result = find_path(k,free_n)
                    n -= free_n - find_result
                    if find_result != free_n:
                        match_right[j][k] -= free_n - find_result
                        if match_right[j][k] == 0:
                            del match_right[j][k]
                        if i in match_right[j]:
                            match_right[j][i] += free_n - find_result
                        else:
                            match_right[j][i] = free_n - find_result
                        if n == 0:
                            return 0
                        free_n = find_result
                        if free_n == 0:
                            break
                    else:
                        break
        else:
            if slack_right[j] > gap: 
                slack_right[j] = gap
    visit_left[i] = True
    return n


def KM():
    global adj_matrix
    global number_left
    global number_right
    global label_left
    global label_right
    global match_right
    global visit_left
    global visit_right
    global slack_right
    global last_v_l
    global last_v_r
    
    for i in range(N):
        slack_right = [np.inf for i in range(W)]
        while True:
            visit_left = [False for i in range(N)]
            visit_right = [False for i in range(W)]
        
            n = number_left[i]
            out = False
            while True:
                find_result = find_path(i,n)
                if find_result == 0:    
                    out = True
                    break
                elif find_result != n:
                    n = find_result
                else:
                    number_left[i] = n
                    break
            if out == True: break
            d = np.inf
            for j, slack in enumerate(slack_right):
                if not visit_right[j] and slack < d:
                    d = slack
            last_v_l = []
            last_v_r = []
            for k in range(N):
                if visit_left[k]: 
                    label_left[k] -= d
                    last_v_l.append(k)
            for n in range(W):
                if visit_right[n]: 
                    label_right[n] += d
                    last_v_r.append(n)
    res = 0
    res_s = 0
    res_ad = 0
    for j in range(W):
        for k in match_right[j]:
            if j != W-1 and k != N-1:
                res_s += adj_matrix[k][j]*match_right[j][k]
            else:
                res_ad += adj_matrix[k][j]*match_right[j][k]
            res += adj_matrix[k][j]*match_right[j][k]
    if res != 0:
        return -res,(-res_s/(-res_s-res_ad))
    else:
        return -res,0
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='')
    parser.add_argument("--dataset", type=str, help='')
    parser.add_argument("--dname", type=str, help='')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
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
    dataset_name = args.dname

    
    retrieve_folder = config_class.checkpoint_dir + '/retrieve'
    yao_folder = config_class.checkpoint_dir + '/yao'

    opop = get_OD_matrix(retrieve_folder+'/retr_trajs_{}.pkl'.format(dataset_name))
    ospop = get_OD_matrix(DatasetConfig.dataset_folder + 
                          'train/lonlat/{}_ground.pkl'.format(DatasetConfig.dataset_prefix))
    gridsize = len(opop)
    logging.info("gridsize:"+str(gridsize))

    s_time = time.time()
    logging.info("[pop/spop] start")
    pop = np.zeros((gridsize,gridsize,gridsize,gridsize))
    spop = np.zeros((gridsize,gridsize,gridsize,gridsize))
    pop_sum = 0
    spop_sum = 0
    for a in range(gridsize):
        for b in range(gridsize):
            for c in range(gridsize):
                for d in range(gridsize):
                    pop[a][b][c][d] += opop[a][b][c][d]*10000
                    pop_sum += opop[a][b][c][d]
                    spop[a][b][c][d] += ospop[a][b][c][d]*10000
                    spop_sum += ospop[a][b][c][d]
    time_1 = time.time()

    logging.info("[npop/nspop] start, time:{:.2f}s".format(time_1-s_time))
    n_pop = np.zeros((gridsize,gridsize,gridsize,gridsize))
    n_spop = np.zeros((gridsize,gridsize,gridsize,gridsize))
    for a in range(gridsize):
        for b in range(gridsize):
            for c in range(gridsize):
                for d in range(gridsize):
                    n_pop[a][b][c][d] = int(pop[a][b][c][d]/pop_sum)
                    n_spop[a][b][c][d] = int(spop[a][b][c][d]/spop_sum)
    time_2 = time.time()


    logging.info("[KM] start, time:{:.2f}s".format(time_2-time_1))
    rs = [0,0,0,0,0,0,0]
    for turn in range(6):
        if turn == 0:
            A_m = pop
            B_m = spop
        if turn == 1:
            A_m = pop
            B_m = np.zeros((gridsize,gridsize,gridsize,gridsize))
        if turn == 2:
            A_m = spop
            B_m = np.zeros((gridsize,gridsize,gridsize,gridsize))
        if turn == 3:
            A_m = n_pop
            B_m = n_spop
        if turn == 4:
            A_m = n_pop
            B_m = np.zeros((gridsize,gridsize,gridsize,gridsize))
        if turn == 5:
            A_m = n_spop
            B_m = np.zeros((gridsize,gridsize,gridsize,gridsize))
        OM = np.zeros((gridsize,gridsize,gridsize,gridsize))
        ON = np.zeros((gridsize,gridsize,gridsize,gridsize))
        for a in range(gridsize):
            for b in range(gridsize):
                for c in range(gridsize):
                    for d in range(gridsize):
                        OM[a][b][c][d] = A_m[a][b][c][d]-min(A_m[a][b][c][d],B_m[a][b][c][d])
                        ON[a][b][c][d] = B_m[a][b][c][d]-min(A_m[a][b][c][d],B_m[a][b][c][d])

        M_set = []
        N_set = []
        for ox in range(gridsize):
            for oy in range(gridsize):
                for dx in range(gridsize):
                    for dy in range(gridsize):          
                        if OM[ox][oy][dx][dy] != 0:
                            M_set.append([ox,oy,dx,dy])
                        if ON[ox][oy][dx][dy] != 0:
                            N_set.append([ox,oy,dx,dy])
        N = len(M_set)+1
        W = len(N_set)+1
        if W > N:
            TMP = N
            N = W
            W = TMP
            TMP_set = M_set
            M_set = N_set
            N_set = TMP_set
            TMP_O = OM
            OM = ON
            ON = TMP_O

        adj_matrix = np.zeros((N,W))

        for i in range(N-1):
            for j in range(W-1):
                adj_matrix[i,j] = 0#-2
                for k in range(4):
                    adj_matrix[i,j] += -abs(M_set[i][k]-N_set[j][k])
            adj_matrix[i,W-1] = min(-(abs(M_set[i][2]-M_set[i][0])+abs(M_set[i][3]-M_set[i][1])),-1)
        for j in range(W-1):
            adj_matrix[N-1,j] = min(-(abs(N_set[j][2]-N_set[j][0])+abs(N_set[j][3]-N_set[j][1])),-1)
        adj_matrix[N-1,W-1] = 0

        number_left = np.zeros((N))
        number_right = np.zeros((W))

        M_total_number = 0
        N_total_number = 0
        for i in range(N-1):
            tmp_n = OM[M_set[i][0]][M_set[i][1]][M_set[i][2]][M_set[i][3]]
            number_left[i] = tmp_n
            M_total_number += tmp_n
        for i in range(W-1):
            tmp_n = ON[N_set[i][0]][N_set[i][1]][N_set[i][2]][N_set[i][3]]
            number_right[i] = tmp_n
            N_total_number += tmp_n
        number_left[N-1] = N_total_number
        number_right[W-1] = M_total_number

        label_left = np.max(adj_matrix, axis=1)  
        label_right = np.zeros(W)  

        match_right = [{} for i in range(W)]

        visit_left = [False for i in range(N)]
        visit_right = [False for i in range(W)]
        slack_right = [np.inf for i in range(W)]

        last_v_l = []
        last_v_r = []

        KM_rs = KM()
        
        rs[turn] = KM_rs[0]/10000
        if turn == 0:
            rs[6] = KM_rs[1]
            logging.info("shift proportion(SP):"+str(KM_rs[1]))

    logging.info('[Save] begin, time:{:.2f}s'.format(time.time()-time_2))
    logging.info("mass difference(MD):"+str(rs[0]))
    logging.info("mass similarity(NMD):"+str(1-rs[0]/(rs[1]+rs[2])))
    logging.info("incluveness(NMA):"+str(1-math.acos(round(rs[1]*rs[1]+rs[2]*rs[2]-rs[0]*rs[0],8)/round(2*rs[1]*rs[2],8))/math.acos(-1)))
    logging.info("structure similarity(RRNSA):"+str(1-math.acos(round(rs[4]*rs[4]+rs[5]*rs[5]-rs[3]*rs[3],8)/round(2*rs[4]*rs[5],8))/math.acos(-1)))
    with open(yao_folder+'/MD_NMD_{}.csv'.format(dataset_name), 'w') as f:
        f.write("mass difference(MD):"+str(rs[0])+"\n")
        f.write("mass similarity(NMD):"+str(1-rs[0]/(rs[1]+rs[2]))+"\n")
        f.write("incluveness(NMA):"+str(1-math.acos(round(rs[1]*rs[1]+rs[2]*rs[2]-rs[0]*rs[0],8)/round(2*rs[1]*rs[2],8))/math.acos(-1))+"\n")
        f.write("structure similarity(RRNSA):"+str(1-math.acos(round(rs[4]*rs[4]+rs[5]*rs[5]-rs[3]*rs[3],8)/round(2*rs[4]*rs[5],8))/math.acos(-1))+"\n")
        f.write("shift proportion(SP):"+str(rs[6])+"\n")
    f.close()
    logging.info('[Save] to:'+yao_folder+'/MD_NMD_{}.csv'.format(dataset_name))
