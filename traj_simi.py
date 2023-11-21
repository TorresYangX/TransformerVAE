import numpy as np
import pandas as pd
import time
from fastdtw import fastdtw
import os
import argparse
from tqdm import trange

def loadData(data, BATCH_SIZE):
    trajectories = pd.read_csv(data, header = None)
    trajectories.columns = ['time', 'longitude', 'latitude', 'id']
    resid = int(((len(trajectories) / 60) // BATCH_SIZE) * BATCH_SIZE * 60)
    trajectories = trajectories[:resid]
    return trajectories

def loadHistoricalDataByDay(dataPath, BATCH_SIZE, targetData, history):
    day = targetData.split('/')[-1].split('_')[0] 
    hour = targetData.split('/')[-1].split('_')[1].split('.')[0] # 7
    historicalTrajectories = pd.DataFrame()
    if int(hour) == 0:
        for i in range(int(day)-history, int(day)-history+1):
            for j in range(24):
                file = '{}_{}.csv'.format(i, j)
                if os.path.exists(dataPath + file):
                    temp =loadData(dataPath + file, BATCH_SIZE)
                    historicalTrajectories = pd.concat([historicalTrajectories, temp], axis=0)
    else:
        for i in range(int(day)-history, int(day)-history+1):
            for j in range(24):
                file = '{}_{}.csv'.format(i, j)
                if os.path.exists(dataPath + file):
                    temp = loadData(dataPath + file, BATCH_SIZE)
                    historicalTrajectories = pd.concat([historicalTrajectories, temp], axis=0)
        for i in range(int(day)-history+1, int(day)-history+1+1):
            for j in range(int(hour)):
                file = '{}_{}.csv'.format(i, j)
                if os.path.exists(dataPath + file):
                    temp = loadData(dataPath + file, BATCH_SIZE)
                    historicalTrajectories = pd.concat([historicalTrajectories, temp], axis=0)
    historicalTrajectories = historicalTrajectories.reset_index(drop=True)
    return historicalTrajectories

def SED(point1, point2):
    return np.sqrt(np.sum(np.square([point1[a] - point2[a] for a in range(len(point1))])))


def DTW(vec1, vec2):
    d = np.zeros([len(vec1) + 1, len(vec2) + 1])
    d[:] = np.exp(10)
    d[0, 0] = 0
    for i in range(1, d.shape[0]):
        for j in range(1, d.shape[1]):
            dist = SED(vec1[i - 1], vec2[j - 1])
            d[i, j] = dist + min(d[i - 1, j - 1], d[i - 1, j], d[i, j - 1])
    return d[-1][-1]


def subcost(loc1, loc2, thres):
    if ((abs(loc1[0] - loc2[0]) <= thres) and (abs(loc1[1] - loc2[1]) <= thres)):
        return 0
    return 1


def EDR(vec1, vec2, thres):
    d = np.zeros([len(vec1) + 1, len(vec2) + 1])
    d[:] = np.inf
    for i in range(d.shape[0]):
        d[i, 0] = i
    for i in range(d.shape[1]):
        d[0, i] = i
    for i in range(1, d.shape[0]):
        for j in range(1, d.shape[1]):
            d[i, j] = min(d[i - 1, j - 1] + subcost(vec1[i - 1], vec2[j - 1], thres), d[i - 1, j] + 1, d[i, j - 1] + 1)
    return d[-1][-1]

def subcost_time(loc1, loc2, thres, time1, time2, time_thres):
    if ((abs(loc1[0] - loc2[0]) <= thres) and (abs(loc1[1] - loc2[1]) <= thres) and (abs(time1 - time2) <= time_thres)):
        return True
    return False


def LCSS(vec1, vec2, thres, time_thres):
    d = np.zeros([len(vec1) + 1, len(vec2) + 1])
    for i in range(1, d.shape[0]):
        for j in range(1, d.shape[1]):
            if subcost_time(vec1[i - 1], vec2[j - 1], thres, i, j, time_thres):
                d[i, j] = 1 + d[i - 1, j - 1]
            else:
                d[i, j] = max(d[i - 1, j], d[i, j - 1])
    return d[-1][-1]


def EDwP_rep(e1_p1, e1_p2, e2_p1, e2_p2):
    return SED(e1_p1, e2_p1) + SED(e1_p2, e2_p2)


def EDwP_cov(e1_p1, e1_p2, e2_p1, e2_p2):
    return SED(e1_p1, e1_p2) + SED(e2_p1, e2_p2)


def EDwP_proj(e_p1, e_p2, p):
    if EDwP_equals(e_p1, e_p2):
        return e_p1

    dot_product_temp = EDwP_dot_product(e_p1, e_p2, p)
    len_2 = np.square(e_p2[0] - e_p1[0]) + np.square(e_p2[1] - e_p1[1])

    x = e_p1[0] + float(dot_product_temp * (e_p2[0] - e_p1[0])) / len_2
    y = e_p1[1] + float(dot_product_temp * (e_p2[1] - e_p1[1])) / len_2
    return [x, y]


def EDwP_dot_product(e_p1, e_p2, p):
    shift_e, shift_p = [0, 0], [0, 0]
    for i in range(0, 2):
        shift_e[i] = e_p2[i] - e_p1[i]
        shift_p[i] = p[i] - e_p1[i]
    dot_product_ret = 0
    for i in range(0, 2):
        dot_product_ret += (shift_e[i] * shift_p[i])
    return dot_product_ret


def EDwP_rest(l):
    if len(l) > 0:
        return l[1:]


def EDwP_equals(a, b):
    for i in range(0, 2):
        if a[i] != b[i]:
            return False
    return True


def EDwP(vec1, vec2):
    vec1 = list(vec1)
    vec2 = list(vec2)
    total_cost_edwp = 0
    if len(vec1) == 0 and len(vec2) == 0:
        return 0
    if len(vec1) == 0 or len(vec2) == 0:
        return np.inf

    flag = False
    while not flag:
        replacement, coverage = 0, 0
        if len(vec1) == 1 and len(vec2) > 1:
            e_p1, e_p2 = vec2[0], vec2[1]
            p = vec1[0]
            replacement = EDwP_rep(e_p1, e_p2, p, p)
            coverage = EDwP_cov(e_p1, e_p2, p, p)
        elif len(vec2) == 1 and len(vec1) > 1:
            e_p1, e_p2 = vec1[0], vec1[1]
            p = vec2[0]
            replacement = EDwP_rep(e_p1, e_p2, p, p)
            coverage = EDwP_cov(e_p1, e_p2, p, p)
        elif len(vec1) > 1 and len(vec2) > 1:
            e1_p1, e1_p2 = vec1[0], vec1[1]
            e2_p1, e2_p2 = vec2[0], vec2[1]
            p_ins_e1 = EDwP_proj(e1_p1, e1_p2, e2_p2)
            p_ins_e2 = EDwP_proj(e2_p1, e2_p2, e1_p2)

            replace_e1 = EDwP_rep(e1_p1, p_ins_e1, e2_p1, e2_p2)
            replace_e2 = EDwP_rep(e2_p1, p_ins_e2, e1_p1, e1_p2)
            cover_e1 = EDwP_cov(e1_p1, p_ins_e1, e2_p1, e2_p2)
            cover_e2 = EDwP_cov(e2_p1, p_ins_e2, e1_p1, e1_p2)

            if cover_e1 * replace_e1 < cover_e2 * replace_e2:
                replacement = replace_e1
                coverage = cover_e1

                if not EDwP_equals(p_ins_e1, e1_p1) and not EDwP_equals(p_ins_e1, e1_p2):
                    vec1.insert(1, p_ins_e1)
            else:
                replacement = replace_e2
                coverage = cover_e2
                if not EDwP_equals(p_ins_e2, e2_p1) and not EDwP_equals(p_ins_e2, e2_p2):
                    vec2.insert(1, p_ins_e2)
        else:
            flag = True
        vec1, vec2 = EDwP_rest(vec1), EDwP_rest(vec2)
        total_cost_edwp += (replacement * coverage)
    return total_cost_edwp


def traj_length(traj):
    sum_len = 0
    for i in range(len(traj) - 1):
        e_p1, e_p2 = traj[i], traj[i + 1]
        sum_len += SED(e_p1, e_p2)
    return sum_len


def main(function_map, method_, history, targetData, historicalData, BATCH_SIZE=16):
    targetTraj = loadData(targetData, BATCH_SIZE)
    targetDataNum = int(len(targetTraj)/60)
    print(targetDataNum)
    historicalTraj = loadHistoricalDataByDay(historicalData, BATCH_SIZE, targetData, history)
    historicalDataNum = int(len(historicalTraj)/60)
    print(historicalDataNum)
    targetTrajectories_ = np.array([list(a) for a in zip(targetTraj.latitude.tolist(), targetTraj.longitude.tolist())]).reshape((int(len(targetTraj)/60), 60, 2))
    historicalTrajectories_ = np.array([list(a) for a in zip(historicalTraj.latitude.tolist(), historicalTraj.longitude.tolist())]).reshape((int(len(historicalTraj)/60), 60, 2))   

    outputPath = '../results/{}/'.format(method_)
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    outputFile = outputPath + 'history_{}.npy'.format(str(history))
    
    container = np.zeros((targetDataNum, historicalDataNum))
    method = function_map[method_]
    if method_ == 'DTW':
        for i in trange(targetDataNum):
            print('DTW calculating {} / {}...'.format(i + 1, targetDataNum), time.ctime())
            for j in range(historicalDataNum):
                container[i, j], _ = method(targetTrajectories_[i, :, :], historicalTrajectories_[j, :, :])
    
    elif method_ == 'EDR':
        for i in trange(targetDataNum):
            print('EDR calculating {} / {}...'.format(i + 1, targetDataNum), time.ctime())
            for j in range(historicalDataNum):
                container[i, j] = method(targetTrajectories_[i, :, :], historicalTrajectories_[j, :, :], 0.25)
    elif method_ == 'LCSS':
        for i in trange(targetDataNum):
            print('LCSS calculating {} / {}...'.format(i + 1, targetDataNum), time.ctime())
            for j in range(historicalDataNum):
                container[i, j] = method(targetTrajectories_[i, :, :], historicalTrajectories_[j, :, :], 0.25, 5)
    elif method_ == 'EDwP':
        for i in trange(targetDataNum):
            print('EDwP calculating {} / {}...'.format(i + 1, targetDataNum), time.ctime())
            for j in range(historicalDataNum):
                container[i, j] = method(targetTrajectories_[i, :, :], historicalTrajectories_[j, :, :])
    else:
        raise('FUNCTION ERROR!')
    np.save(outputFile, container)
    return 0
    
if __name__ == '__main__':
    data = pd.read_csv('../data/Experiment/experiment_data_before_time/8_7.csv', header=None)
    data.columns = ['time', 'lon', 'lat', 'id']
    choose1 = data[data.id == 89]
    choose2 = data[data.id == 96] 
    traj1 = [list(a) for a in (zip(choose1.lat.tolist(), choose1.lon.tolist()))]
    traj2 = [list(a) for a in (zip(choose2.lat.tolist(), choose2.lon.tolist()))]
    print(len(traj1), len(traj2))

    # DTW
    ss = time.time()
    DTW_score = DTW(traj1, traj2)
    tt = time.time()

    print(tt-ss, 'DTW')
    distance, _ = fastdtw(traj1, traj2)
    print(time.time()-tt, 'fast DTW')
    

    # EDR
    # note that EDR requires to normalize trajectories first
    thres = 0.25
    ss = time.time()
    EDR_score = EDR(traj1, traj2, thres)
    print(time.time()-ss, 'EDR')

    # LCSS
    # not achieve S2 LCSS
    thres = 0.25
    time_thres = 5
    ss = time.time()
    LCSS_score = LCSS(traj1, traj2, thres, time_thres)
    print(time.time() - ss, 'LCSS')
    S1_LCSS = LCSS_score / min(len(traj1), len(traj2))

    # EDwP
    ss = time.time()
    EDwP_score = EDwP(traj1, traj2)
    print(time.time() - ss, 'EDwP')
    EDwP_score_avg = EDwP_score / (traj_length(traj1) + traj_length(traj2))
    print('  DTW    similar: ', DTW_score)
    print('  EDR    similar: ', EDR_score)
    print('  LCSS   similar: ', LCSS_score)
    print(' S1_LCSS similar: ', S1_LCSS)
    print('  EDwP   similar: ', EDwP_score)
    print('EDwP_avg similar: ', EDwP_score_avg)
    
    function_map = {'DTW': fastdtw,
                    'EDR': EDR,
                    'LCSS': LCSS,
                    'EDwP': EDwP}
    
    parse = argparse.ArgumentParser()
    parse.add_argument('-m', '--method_', dest='method_', required=True, type=str)
    parse.add_argument('-hi', '--history', dest='history', required=True, type = int)
    parse.add_argument('-t', '--targetData', dest='targetData', required=True, type = str)
    parse.add_argument('-hd', '--historicalData', dest='historicalData', required=True, type=str)
    parse.add_argument('-b', '--BATCH_SIZE', dest='BATCH_SIZE', required=False, type=int)
    
    args = parse.parse_args()
    main(function_map, args.method_, args.history, args.targetData, args.historicalData, args.BATCH_SIZE)