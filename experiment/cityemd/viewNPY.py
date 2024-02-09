#view npy file
import numpy as np
import pandas as pd
import argparse

dataset = 'Geolife'

def viewNpy(targetData):
    file = np.load(targetData)
    print(file)

def main(args):
    targetData =  '../results/{}/{}/KDTree{}/EMD/emd_.npy'.format(dataset, args.MODEL, args.MODEL)
    viewNpy(targetData)
    file = np.load(targetData)
    mean = np.mean(file)
    logmean = np.log(mean)
    print(logmean)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--MODEL", type=str, default="LCSS", choices=["LCSS", "DTW", "EDR", "EDwP", "VAE", "AE", "NVAE", "Transformer", "t2vec"], required=True)
    args = parser.parse_args()
    main(args)
