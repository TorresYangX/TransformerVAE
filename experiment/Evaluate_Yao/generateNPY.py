import numpy as np
import pandas as pd
import argparse 
import os

lat1 = 39.6
lat2 = 40.2
lon1 = 116
lon2 = 116.8
grid_num = 50
grid_size = max(lon2 - lon1, lat2 - lat1) / grid_num
trajectory_length = 60

def generateNPY(targetFile, saveFilePath):
    if not os.path.exists(os.path.dirname(saveFilePath)):
        os.makedirs(os.path.dirname(saveFilePath))
    df = pd.read_csv(targetFile, header=None)
    data = df.to_numpy()[:, 1:3]
    data = data.reshape(-1, trajectory_length, 2)
    OD_grid = np.zeros((grid_num, grid_num, grid_num, grid_num))
    data[:, :, 0] = np.floor((data[:, :, 0] - lon1) / grid_size)
    data[:, :, 1] = np.floor((data[:, :, 1] - lat1) / grid_size)
    new_data = np.zeros((data.shape[0], trajectory_length-1, 2, 2))
    for i in range(trajectory_length-1):
        new_data[:, i, 0, 0] = data[:, i, 0]
        new_data[:, i, 0, 1] = data[:, i, 1]
        new_data[:, i, 1, 0] = data[:, i+1, 0]
        new_data[:, i, 1, 1] = data[:, i+1, 1]
    for i in range(new_data.shape[0]):
        for j in range(new_data.shape[1]):
            OD_grid[int(new_data[i, j, 0, 0]), int(new_data[i, j, 0, 1]), int(new_data[i, j, 1, 0]), int(new_data[i, j, 1, 1])] += 1
    np.save(saveFilePath, OD_grid)
    
         
    
def main(args):
    if args.MODEL == 'groundTruth':
        targetData =  '../data/beijing/Experiment/groundTruth/groundTruth_8.csv'
        saveFilePath = '../data/beijing/Experiment/groundTruth/OD_MATRIX.npy'
        generateNPY(targetData, saveFilePath)
    else:
        targetData =  '../results/{}/KDTree{}/EMD/retrievedTrajectories.csv'.format(args.MODEL, args.MODEL)
        saveFilePath = '../results/{}/KDTree{}/Evaluate_Yao/OD_MATRIX.npy'.format(args.MODEL, args.MODEL)
        generateNPY(targetData, saveFilePath)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--MODEL", type=str, default="LCSS", choices=["LCSS", "EDR", "EDwP", "DTW","VAE", "AE", "NVAE", "Transformer", "t2vec", "groundTruth"], required=True)
    args = parser.parse_args()
    main(args)