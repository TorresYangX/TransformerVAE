from baseFuncs import *
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import trange


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class IndexEncoder():
    def __init__(self, model, trajectory_length, grid_num, embedding_dim, Batch_size):
        self.model = model
        self.trajectory_length = trajectory_length
        self.grid_num = grid_num
        self.embedding_dim = embedding_dim
        self.Batch_size = Batch_size
    
    def encoding(self, dataPath):
        muFolder = '../results/VAE_nvib/Index/mu/'
        sigmaFolder = '../results/VAE_nvib/Index/sigma/'
        piFolder = '../results/VAE_nvib/Index/pi/'
        alphaFolder = '../results/VAE_nvib/Index/alpha/'
        if not os.path.exists(muFolder):
            os.makedirs(muFolder)
        if not os.path.exists(sigmaFolder):
            os.makedirs(sigmaFolder)
        if not os.path.exists(piFolder):
            os.makedirs(piFolder)
        if not os.path.exists(alphaFolder):
            os.makedirs(alphaFolder)
        for i in trange(2, 9):
            for j in range(24):
                FILE = '{}_{}.npy'.format(i, j)
                if os.path.exists(dataPath+FILE):
                    muFILE = muFolder + 'mu_{}_{}.csv'.format(i, j)
                    sigmaFILE = sigmaFolder + 'sigma_{}_{}.csv'.format( i, j)
                    piFILE = piFolder + 'pi_{}_{}.csv'.format(i, j)
                    alphaFILE = alphaFolder + 'alpha_{}_{}.csv'.format(i, j)
                    x = constructSingleData(dataPath, FILE, self.Batch_size)
                    if x.shape[0] > 0:
                        predict_data = np.array(x)
                        predict_dataset = torch.utils.data.TensorDataset(torch.from_numpy(predict_data))
                        predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=self.Batch_size)
                        self.model.eval()
                        result_mu = []
                        result_sigma = []
                        result_pi = []
                        result_alpha = []
                        for _, x in enumerate(predict_loader):
                            x = x[0].transpose(0,1).to(device)
                            eos = torch.full((1, x.shape[1]), self.grid_num*self.grid_num).to(device)
                            sos = torch.full((1, x.shape[1]), self.grid_num*self.grid_num+1).to(device)
                            src = torch.cat([x, eos], dim=0)
                            tgt = torch.cat([sos, x], dim=0)    
                            src_key_padding_mask = torch.zeros((x.shape[1], self.trajectory_length + 1), dtype=torch.bool).to(device)
                            tgt_key_padding_mask = torch.zeros((x.shape[1], self.trajectory_length + 1), dtype=torch.bool).to(device)
                            # Forward pass
                            outputs_dict = self.model(
                                src,
                                tgt,
                                src_key_padding_mask=src_key_padding_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                            )
                            mu = outputs_dict["mu"].mean(dim=0, keepdim=True) #(1,16,16)
                            logvar = outputs_dict["logvar"].mean(dim=0, keepdim=True) #(1,16,16)
                            pi = outputs_dict["pi"].repeat(1,1,self.embedding_dim) #(61,16,16)
                            alpha = outputs_dict["alpha"].repeat(1,1,self.embedding_dim) #(61,16,16)
                            pi = pi.mean(dim=0, keepdim=True) #(1,16,16)
                            alpha = alpha.mean(dim=0, keepdim=True) #(1,16,16)
                            result_mu.append(mu.cpu().detach().numpy())
                            result_sigma.append(logvar.cpu().detach().numpy())
                            result_pi.append(pi.cpu().detach().numpy())
                            result_alpha.append(alpha.cpu().detach().numpy())                   
                        result_mu = np.concatenate(result_mu, axis = 1)
                        result_sigma = np.concatenate(result_sigma, axis = 1)
                        result_pi = np.concatenate(result_pi, axis = 1)
                        result_alpha = np.concatenate(result_alpha, axis = 1)
                        result_mu =  result_mu.transpose(1,0,2).reshape(result_mu.shape[1], -1)
                        result_sigma = result_sigma.transpose(1,0,2).reshape(result_sigma.shape[1], -1)
                        result_pi = result_pi.transpose(1,0,2).reshape(result_pi.shape[1], -1)
                        result_alpha = result_alpha.transpose(1,0,2).reshape(result_alpha.shape[1], -1)
                        parameteroutput(result_mu, muFILE)
                        parameteroutput(result_sigma, sigmaFILE)
                        parameteroutput(result_pi, piFILE)
                        parameteroutput(result_alpha, alphaFILE)

                        