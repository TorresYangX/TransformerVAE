from baseFuncs import *
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import trange
from t2vec import EncoderDecoder


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

m0SavePath = '../results/t2vec/m0.pt'
m1SavePath = '../results/t2vec/m1.pt'
grid_num = 50
Batch_size = 16
vocab_size = 2502
embedding_dim = 64
hidden_dim = 16
dropout = 0.1

class IndexEncoder():
    def __init__(self, m0, m1, grid_num, embedding_dim, Batch_size):
        self.m0 = m0
        self.m1 = m1
        self.grid_num = grid_num
        self.embedding_dim = embedding_dim
        self.Batch_size = Batch_size
    
    def encoding(self, dataPath):
        indexFolder = '../results/t2vec/Index/prob/'
        if not os.path.exists(indexFolder):
            os.makedirs(indexFolder)
        for i in trange(2, 9):
            for j in range(24):
                FILE = '{}_{}.npy'.format(i, j)
                if os.path.exists(dataPath+FILE):
                    indexFILE = indexFolder + 'prob_{}_{}.csv'.format(i, j)
                    x = constructSingleData(dataPath, FILE, self.Batch_size)
                    if x.shape[0] > 0:
                        predict_data = np.array(x)
                        predict_dataset = torch.utils.data.TensorDataset(torch.from_numpy(predict_data))
                        predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=self.Batch_size)
                        self.m0.eval()
                        self.m1.eval()
                        result_index = []
                        for _, x in enumerate(predict_loader):
                            x = x[0].transpose(0,1).to(device)
                            eos = torch.full((1, x.shape[1]), self.grid_num*self.grid_num).to(device)
                            sos = torch.full((1, x.shape[1]), self.grid_num*self.grid_num+1).to(device)
                            src = torch.cat([x, eos], dim=0)
                            tgt = torch.cat([sos, x], dim=0)    
                            src = src.long()
                            lengths = torch.full((1, x.shape[1]), x.shape[0]).to(device)
                            tgt = tgt.long()
                            _, decoder_h0 = self.m0(src, lengths, tgt)
                            index = decoder_h0.mean(dim=0)
                            result_index.append(index.cpu().detach().numpy())
                        result_index = np.concatenate(result_index, axis = 0)
                        parameteroutput(result_index, indexFILE)
                        
                        
                        
if __name__ == '__main__':
    # load model
    m0 = EncoderDecoder(vocab_size,
                        embedding_dim,
                        hidden_dim,
                        num_layers=3,
                        dropout=dropout,
                        bidirectional=True).to(device)
    m0.load_state_dict(torch.load(m0SavePath))
    m1 = nn.Sequential(nn.Linear(hidden_dim, vocab_size),
                       nn.LogSoftmax(dim=1)).to(device)
    m1.load_state_dict(torch.load(m1SavePath))
    # load data
    dataPath = '../data/beijing/Experiment/experimentGridData/'
    # encoding
    indexEncoder = IndexEncoder(m0, m1, grid_num, embedding_dim, Batch_size)
    indexEncoder.encoding(dataPath)

                        