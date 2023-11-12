#view npy file
import numpy as np
import pandas as pd

def viewNpy(targetData):
    file = np.load(targetData)
    print(file)

if __name__ == '__main__':
    targetData =  '../results/LCSS/KDTreeLCSS/EMD/emd_.npy'
    viewNpy(targetData)
    file = np.load(targetData)
    mean = np.mean(file)
    logmean = np.log(mean)
    print(logmean)
