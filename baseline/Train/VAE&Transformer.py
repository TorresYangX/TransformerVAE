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

from baseFuncs import *

from baseline.models.VAE import VAE
from baseline.models.VAE import AE
from baseline.models.VAE import Transformer


BATCH_SIZE = 16
grid_num = 50
vocab_size = grid_num * grid_num + 2
dropout = 0.1
learning_rate = 1e-3
embedding_dim = 64
hidden_dim = 32
latent_dim = 16
MAX_EPOCH = 300

NUM_HEADS = 8
NUM_LAYERS = 6
DIM_FORWARD = 512


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def train(model, train_loader, optimizer, trajectory_length, jargs):
    model.train()
    train_loss = 0
    for _, x in enumerate(train_loader):
        if args.MODEL == 'VAE' or args.MODEL == "AE":
            x = x[0].to(device)
            input_dict = {
                "x": x,
            }
            tgt = x
        else:
            x = x[0].transpose(0,1).to(device)
            eos = torch.full((1, x.shape[1]), grid_num*grid_num).to(device) # eos = 2500
            sos = torch.full((1, x.shape[1]), grid_num*grid_num+1).to(device) # sos = 2501
            src = torch.cat([x, eos], dim=0)
            tgt = torch.cat([sos, x], dim=0)

            src_key_padding_mask = torch.zeros((x.shape[1], trajectory_length + 1), dtype=torch.bool).to(device)
            tgt_key_padding_mask = torch.zeros((x.shape[1], trajectory_length + 1), dtype=torch.bool).to(device)

            input_dict = {
                "src": src,
                "tgt": tgt,
                "src_key_padding_mask": src_key_padding_mask,
                "tgt_key_padding_mask": tgt_key_padding_mask,
            }

        optimizer.zero_grad()
        dict = model(**input_dict)
        loss = model.loss_fn(targets=tgt, **dict)["Loss"]
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(train_loader.dataset)

def test(model, test_loader, trajectory_length, args):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for _, x in enumerate(test_loader):
            if args.MODEL == 'VAE' or args.MODEL == "AE":
                x = x[0].to(device)
                input_dict = {
                    "x": x,
                }
                tgt = x
            else:
                x = x[0].transpose(0,1).to(device)
                eos = torch.full((1, x.shape[1]), grid_num*grid_num).to(device)
                sos = torch.full((1, x.shape[1]), grid_num*grid_num+1).to(device)
                src = torch.cat([x, eos], dim=0)
                tgt = torch.cat([sos, x], dim=0)

                src_key_padding_mask = torch.zeros((x.shape[1], trajectory_length + 1), dtype=torch.bool).to(device)
                tgt_key_padding_mask = torch.zeros((x.shape[1], trajectory_length + 1), dtype=torch.bool).to(device)

                input_dict = {
                    "src": src,
                    "tgt": tgt,
                    "src_key_padding_mask": src_key_padding_mask,
                    "tgt_key_padding_mask": tgt_key_padding_mask,
                }

            dict = model(**input_dict)
            test_loss += model.loss_fn(targets=tgt, **dict)["Loss"].item()
    return test_loss / len(test_loader.dataset)


def trainModel(trainFilePath, modelSavePath, trainlogPath, trajectory_length, args):
    x = constructTrainingData(trainFilePath, BATCH_SIZE)
    dataSet = torch.utils.data.TensorDataset(torch.from_numpy(x))
    val_size = int(0.2 * len(dataSet))

    if args.MODEL == 'VAE' or args.MODEL == "AE":
        model = {
            "VAE": VAE,
            "AE": AE,
        }[args.MODEL](embedding_dim, hidden_dim, latent_dim, vocab_size, BATCH_SIZE, trajectory_length).to(device)
    elif args.MODEL == 'Transformer':
        model = {
            "Transformer": Transformer,
        }[args.MODEL](latent_dim, NUM_HEADS, NUM_LAYERS, DIM_FORWARD, dropout, vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    
    logger = get_logger(trainlogPath)
    train_loss_list = []
    test_loss_list = []
    print("Start training...")
    for epoch in trange(MAX_EPOCH):
        train_dataset, test_dataset = torch.utils.data.random_split(dataSet, [len(dataSet) - val_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
        train_loss = train(model, train_loader, optimizer, trajectory_length, args)
        test_loss = test(model, test_loader, trajectory_length, args)
        logger.info('Epoch:[{}/{}]\t Train Loss={:.4f}\t Test Loss={:.4f}'.format(epoch+1 , MAX_EPOCH, train_loss, test_loss ))
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
    print("End training...")
    torch.save(model, modelSavePath)
    plot_loss(train_loss_list, test_loss_list, args)


def encoding(modelPath, dataPath, trajectory_length, args):
    if not args.SSM_KNN:
        indexPath = '../results/{}/Index/'.format(args.MODEL)
        if not os.path.exists(indexPath):
            os.makedirs(indexPath)
    else:
        indexPath = '../SSM_KNN/{}/Index/'.format(args.MODEL)
        if not os.path.exists(indexPath):
            os.makedirs(indexPath)
        dbNUM = dataPath.split('/')[-3]
        indexPath = indexPath + dbNUM + '/'
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
                        for _, src in enumerate(predict_loader):
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
                            prob = dict['prob'][:, 0, :]
                            result_prob.append(prob.cpu().detach().numpy())
                        result_prob = np.concatenate(result_prob, axis = 0)
                        parameteroutput(result_prob, probFILE)
                    elif args.MODEL == "Transformer":
                        result_prob = []
                        for _, x in enumerate(predict_loader):
                            x = x[0].transpose(0,1).to(device)
                            eos = torch.full((1, x.shape[1]), grid_num*grid_num).to(device)
                            sos = torch.full((1, x.shape[1]), grid_num*grid_num+1).to(device)
                            src = torch.cat([x, eos], dim=0)
                            tgt = torch.cat([sos, x], dim=0)

                            src_key_padding_mask = torch.zeros((x.shape[1], trajectory_length + 1), dtype=torch.bool).to(device)
                            tgt_key_padding_mask = torch.zeros((x.shape[1], trajectory_length + 1), dtype=torch.bool).to(device)

                            input_dict = {
                                "src": src,
                                "tgt": tgt,
                                "src_key_padding_mask": src_key_padding_mask,
                                "tgt_key_padding_mask": tgt_key_padding_mask,
                            }

                            dict = model(**input_dict)
                            encoder_ouput = dict['z']
                            prob = encoder_ouput.mean(dim=0, keepdim=True) #(1,16,16)
                            result_prob.append(prob.cpu().detach().numpy())
                        result_prob = np.concatenate(result_prob, axis = 1)
                        result_prob =  result_prob.transpose(1,0,2).reshape(result_prob.shape[1], -1)
                        parameteroutput(result_prob, probFILE)


def main(args):
    trajectory_length = 60
    root = '../results/{}/'.format(args.MODEL)
    if not os.path.exists(root):
        os.makedirs(root)
    save_model = '../results/{}/{}.pt'.format(args.MODEL, args.MODEL)
    trainlog = '../results/{}/trainlog.csv'.format(args.MODEL)
    trainFilePath = '../data/{}/Train/trainGridData/'.format(args.DATASET)
    if args.TASK=="train":
        trainModel(trainFilePath, save_model, trainlog, trajectory_length, args)
    else:
        dataPath = '../data/{}/Experiment/experimentGridData/'.format(args.DATASET)
        encoding(save_model, dataPath, trajectory_length, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--TASK", type=str, default='train', choices=["train","encode"],help="train or encode", required=True)

    parser.add_argument("-d", "--DATASET", type=str, default="beijing", choices=["beijing","MCD"] ,help="dataset", required=True)

    parser.add_argument("-m", "--MODEL", type=str, default="VAE", choices=["VAE", "AE", "Transformer"], required=True)

    args = parser.parse_args()

    main(args)
