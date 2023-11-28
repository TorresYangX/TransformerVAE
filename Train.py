import numpy as np
import pandas as pd
import math
import logging
import os
from tqdm import trange
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from classes.VAE import VAE
from classes.AE import AE


BATCH_SIZE = 16
grid_num = 50
vocab_size = grid_num * grid_num
dropout = 0.1
learning_rate = 0.001
embedding_dim = 64
hidden_dim = 32
latent_dim = 16
MAX_EPOCH = 300


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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


def train(model, train_loader, optimizer):
    model.train()
    train_loss = 0
    for _, x in enumerate(train_loader):
        x = x[0].to(device)
        optimizer.zero_grad()
        dict = model(x)
        loss = model.loss_fn(x=x, **dict)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(train_loader.dataset)

def test(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for _, x in enumerate(test_loader):
            x = x[0].to(device)
            dict = model(x)
            test_loss += model.loss_fn(x=x, **dict).item()
    return test_loss / len(test_loader.dataset)


def trainModel(trainFilePath, modelSavePath, trainlogPath, trajectory_length, args):
    x = constructTrainingData(trainFilePath, BATCH_SIZE)
    # split traindata and testdata
    train_data = x[:int(len(x)*15/16), :]
    test_data = x[int(len(x)*15/16):, :]

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data))
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = {
        "VAE": VAE,
        "AE": AE,
    }[args.MODEL](embedding_dim, hidden_dim, latent_dim, vocab_size, BATCH_SIZE, trajectory_length).to(device)
    optimizer = optim.RMSprop(model.parameters(),lr=learning_rate)
    
    logger = get_logger(trainlogPath)
    train_loss_list = []
    test_loss_list = []
    print("Start training...")
    for epoch in trange(MAX_EPOCH):
        train_loss = train(model, train_loader, optimizer)
        test_loss = test(model, test_loader)
        logger.info('Epoch:[{}/{}]\t Train Loss={:.4f}\t Test Loss={:.4f}'.format(epoch+1 , MAX_EPOCH, train_loss, test_loss ))
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
    print("End training...")
    torch.save(model, modelSavePath)
    x=[i for i in range(len(train_loss_list))]
    figure = plt.figure(figsize=(20, 8), dpi=80)
    plt.plot(x,train_loss_list,label='train_losses')
    plt.plot(x,test_loss_list,label='test_losses')
    plt.xlabel("iterations",fontsize=15)
    plt.ylabel("loss",fontsize=15)
    plt.legend()
    plt.grid()
    plt.savefig('../results/{}/loss_figure.png'.format(args.MODEL))
    plt.show()

def encoding(modelPath, dataPath, args):
    if not args.SSM_KNN:
        indexPath = '../results/{}/Index/'.format(args.MODEL)
    else:
        indexPath = '../SSM_KNN/{}/Index/'.format(args.MODEL)
    if not os.path.exists(indexPath):
        os.makedirs(indexPath)
    muPath = indexPath + 'mu/'
    sigmaPath = indexPath + 'sigma/'
    probPath = indexPath + 'prob/'
    for i in trange(2, 9):
        for j in range(24):
            FILE = '{}_{}.npy'.format(i, j)
            if os.path.exists(dataPath+FILE):
                muFILE = muPath + 'mu_{}_{}.csv'.format(i, j)
                sigmaFILE = sigmaPath + 'sigma_{}_{}.csv'.format(i, j)
                probFILE = probPath + 'prob_{}_{}.csv'.format(i, j)
                x = constructSingleData(dataPath, FILE, BATCH_SIZE)
                if x.shape[0] > 0:
                    predict_data = np.array(x)
                    predict_dataset = torch.utils.data.TensorDataset(torch.from_numpy(predict_data))
                    predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=BATCH_SIZE)
                    model = torch.load(modelPath)
                    model.eval()
                    if args.MODEL == 'VAE':
                        result_mu = []
                        result_sigma = []
                        for idx, src in enumerate(predict_loader):
                            src = src[0].to(device)
                            dict = model(src)
                            mu = dict['mu'][:,0,:]
                            logvar = dict['logvar'][:,0,:]
                            result_mu.append(mu.cpu().detach().numpy())
                            result_sigma.append(logvar.cpu().detach().numpy())
                        result_mu = np.concatenate(result_mu, axis = 0)
                        result_sigma = np.concatenate(result_sigma, axis = 0)
                        parameteroutput(result_mu, muFILE)
                        parameteroutput(result_sigma, sigmaFILE)
                    elif args.MODEL == "AE":
                        result_prob = []
                        for idx, src in enumerate(predict_loader):
                            src = src[0].to(device)
                            dict = model(src)
                            prob = dict['h'][:, 0, :]
                            result_prob.append(prob.cpu().detach().numpy())
                        result_prob = np.concatenate(result_prob, axis = 0)
                        parameteroutput(result_prob, probFILE)


def main(args):
    if not args.SSM_KNN:
        trajectory_length = 60
        root = '../results/{}/'.format(args.MODEL)
        if not os.path.exists(root):
            os.makedirs(root)
        save_model = '../results/{}/{}.pt'.format(args.MODEL, args.MODEL)
        trainlog = '../results/{}/trainlog.csv'.format(args.MODEL)
        trainFilePath = '../data/Train/trainGridData/'
        if args.TASK=="train":
            trainModel(trainFilePath, save_model, trainlog, trajectory_length, args)
        else:
            dataPath = '../data/Experiment/experimentGridData/'
            encoding(save_model, dataPath, args)

    else:
        trajectory_length = 30
        root = '../SSM_KNN/'
        if not os.path.exists(root):
            os.makedirs(root)
        root = '../SSM_KNN/{}/'.format(args.MODEL)
        if not os.path.exists(root):
            os.makedirs(root)
        save_model = '../SSM_KNN/{}/{}.pt'.format(args.MODEL, args.MODEL)
        trainlog = '../SSM_KNN/{}/trainlog.csv'.format(args.MODEL)
        trainFilePath = '../data/Train/SSM_KNN/DataBase/GridData/'
        dataPath_1 = '../data/Experiment/SSM_KNN/DataBase_1/GridData/'
        dataPath_2 = '../data/Experiment/SSM_KNN/DataBase_2/GridData/'
        if args.TASK=="train":
            trainModel(trainFilePath, save_model, trainlog, trajectory_length, args)
        else:
            encoding(save_model, dataPath_1, args)
            encoding(save_model, dataPath_2, args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--TASK", type=str, default='train', choices=["train","encode"],help="train or encode", required=True)

    parser.add_argument("-m", "--MODEL", type=str, default="VAE", choices=["VAE", "AE"], required=True)

    parser.add_argument("-s", "--SSM_KNN", type=bool, default=False, required=True)

    args = parser.parse_args()

    main(args)
