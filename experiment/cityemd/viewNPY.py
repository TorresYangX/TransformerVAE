#view npy file
import numpy as np
import pandas as pd
import argparse

def viewNpy(targetData):
    file = np.load(targetData)
    print(file)

def main(args):
    targetData =  '../results/{}/KDTree{}/EMD/emd_.npy'.format(args.MODEL, args.MODEL)
    viewNpy(targetData)
    file = np.load(targetData)
    mean = np.mean(file)
    logmean = np.log(mean)
    print(logmean)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--MODEL", type=str, default="LCSS", choices=["LCSS", "EDR", "EDwP", "VAE", "AE", "VAE_nvib"], required=True)
    args = parser.parse_args()
    main(args)
