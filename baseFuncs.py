import math
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def constructTrainingData(filePath, BATCH_SIZE):
    x = []
    for file in os.listdir(filePath):
        data = np.load(filePath + file)
        x.extend(data)
    x = np.array(x)
    resid = (x.shape[0] // BATCH_SIZE) * BATCH_SIZE
    x = x[:resid, :, :]
    x = x[:, :, 0] # only use the grid num
    return x

def constructSingleData(filePath, file, BATCH_SIZE):
    data = np.load(filePath + file)
    x = np.array(data)
    resid = (x.shape[0] // BATCH_SIZE) * BATCH_SIZE
    x = x[:resid, :, :]
    return x[:, :,0]
        
def parameteroutput(data, file):
    directory = os.path.dirname(file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    para = pd.DataFrame(data)
    with open(file, mode = 'w') as f:
        para.to_csv(f, index = False, header = None)

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def plot_loss(train_loss_list, test_loss_list, args):
    x=[i for i in range(len(train_loss_list))]
    figure = plt.figure(figsize=(20, 8), dpi=80)
    plt.plot(x,train_loss_list,label='train_losses')
    plt.plot(x,test_loss_list,label='test_losses')
    plt.xlabel("iterations",fontsize=15)
    plt.ylabel("loss",fontsize=15)
    plt.legend()
    plt.grid()
    if not args.SSM_KNN:
        plt.savefig('../results/{}/loss_figure.png'.format(args.MODEL))
    else:
        plt.savefig('../SSM_KNN/{}/loss_figure.png'.format(args.MODEL))
    plt.show()