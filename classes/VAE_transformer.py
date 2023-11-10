import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import math
import logging
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

grid_num = 50
vocab_size = grid_num * grid_num
Batch_size = 64
trajectory_length = 60
embedding_dim = 512
latent_dim = 1024
MAX_EPOCH = 2000

# same
def constructTrainingData(filePath, BATCH_SIZE):
    x = []
    for file in os.listdir(filePath):
        data = np.load(filePath + file)
        x.append(data)
    x = np.concatenate(x, axis = 0)
    resid = (x.shape[0] // BATCH_SIZE) * BATCH_SIZE
    x = x[:resid, :, :]
    x = x[:, :, 0] # only use the grid num
    return x

def constructSingleData(filePath, file):
    data = np.load(filePath + file)
    return data[:, :,0]
        

def parameteroutput(para, file):
    para = pd.DataFrame(para)
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

class TokenEmbedding(nn.Module):
    def __init__(self):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

    def forward(self, x):
        return self.embedding(x)

class PositionEmbedding(nn.Module):
    def __init__(self,max_len=trajectory_length):
        super(PositionEmbedding, self).__init__()
        self.max_len = max_len
        
    def forward(self, data_length):
        positional_encoding = np.zeros((self.max_len,embedding_dim))
        for pos in range(positional_encoding.shape[0]):
            for i in range(positional_encoding.shape[1]):
                positional_encoding[pos][i] = math.sin(pos/(10000**(2*i/embedding_dim))) if i % 2 == 0 else math.cos(pos/(10000**(2*i/embedding_dim)))
        return torch.from_numpy(np.repeat(positional_encoding[np.newaxis,:,:],data_length,axis=0)).to(device)
    
class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, dim_feedforward=2048, dropout=0.1, activation='relu', batch_first=True, device=device) 
        self.TransformerEncoder = nn.TransformerEncoder(self.TransformerEncoderLayer, num_layers=6)
        
    def forward(self, src):
        return self.TransformerEncoder(src)
    

class TransformerDecoder(nn.Module):
    def __init__(self):
        super(TransformerDecoder, self).__init__()
        self.TransformerDecoderLayer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=8, dim_feedforward=2048, dropout=0.1, activation='relu', batch_first=True, device=device) 
        self.TransformerDecoder = nn.TransformerDecoder(self.TransformerDecoderLayer, num_layers=6)

    def forward(self, tgt, memory, tgt_mask):
        return self.TransformerDecoder(tgt, memory, tgt_mask)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.TokenModel = TokenEmbedding()
        self.PositionModel = PositionEmbedding()
        self.Encoder = TransformerEncoder()
        self.Decoder = TransformerDecoder()
        self.mu_linear = nn.Linear(embedding_dim, latent_dim)
        self.logvar_linear = nn.Linear(embedding_dim, latent_dim)
        self.z_linear = nn.Linear(latent_dim, embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.softmax = nn.Softmax(dim=2)
        self.softsign = nn.Softsign()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def loss_fn(self, x_hat, x, mu, logvar):
        BCE = torch.mean((x_hat - x) ** 2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + 0. * KLD
        
    def forward(self, src):
        Token = self.TokenModel(src.to(torch.int64).to(device))
        Position = self.PositionModel(Token.shape[0])
        embeddingOutput = (Token + Position).to(torch.float32)
        encoderOuput = self.Encoder(embeddingOutput)
        x = encoderOuput[:, -1, :]
        mu = self.softsign(self.mu_linear(x))
        logvar = self.softsign(self.logvar_linear(x))
        z = self.reparameterize(mu, logvar)
        z = self.softsign(self.z_linear(z))
        memory = z.unsqueeze(1).repeat(1, trajectory_length, 1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(trajectory_length).to(device)
        decoderOutput = self.Decoder(encoderOuput, memory, tgt_mask)
        possibleMat = self.fc(decoderOutput)
        possibleMat = self.softmax(possibleMat)
        # greedy decoding
        x = torch.zeros(possibleMat.shape[0], trajectory_length).to(device)
        for batch_num in range(possibleMat.shape[0]):
            for node in range(trajectory_length):
                x[batch_num][node] = torch.argmax(possibleMat[batch_num][node])
        return x, mu, logvar
    
def train(model, train_loader, optimizer):
    model.train()
    train_loss = 0
    for _, x in enumerate(train_loader):
        x = x[0].to(device)
        optimizer.zero_grad()
        x_hat, mu, logvar = model(x)
        loss = model.loss_fn(x_hat, x, mu, logvar)
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
            x_hat, mu, logvar = model(x)
            test_loss += model.loss_fn(x_hat, x, mu, logvar).item()
    return test_loss / len(test_loader.dataset)



def main():
    # define path
    save_model = '../small_results/VAE_transformer_new/VAE_transformer.pt'
    trainlog = '../small_results/VAE_transformer_new/trainlog.csv'

    trainFilePath = '../small_data/GridData/'
    queryFilePath = '../small_data/queryGridData/'

    # # train
    # x = constructTrainingData(trainFilePath, Batch_size)

    # train_data = np.array(x)
    # test_data = np.array(x)

    # train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data))
    # test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data))

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Batch_size)

    # model = VAE().to(device)

    # optimizer = optim.RMSprop(model.parameters(),lr=0.0001)

    # logger = get_logger(trainlog)
    # train_loss_list = []
    # test_loss_list = []
    # for epoch in range(MAX_EPOCH):
    #     train_loss = train(model, train_loader, optimizer)
    #     test_loss = test(model, test_loader)
    #     logger.info('Epoch:[{}/{}]\t Train Loss={:.4f}\t Test Loss={:.4f}'.format(epoch+1 , MAX_EPOCH, train_loss, test_loss ))
    #     train_loss_list.append(train_loss)
    #     test_loss_list.append(test_loss)
    # x=[i for i in range(len(train_loss_list))]
    # figure = plt.figure(figsize=(20, 8), dpi=80)
    # plt.plot(x,train_loss_list,label='train_losses')
    # plt.plot(x,test_loss_list,label='test_losses')
    # plt.xlabel("iterations",fontsize=15)
    # plt.ylabel("loss",fontsize=15)
    # plt.legend()
    # plt.grid()
    # plt.savefig('../small_results/VAE_transformer_new/loss_figure.png')
    # plt.show()

    # # save model
    # torch.save(model, save_model)

    # # encoding history
    # if not os.path.exists('../small_results/VAE_transformer_new/Index/History/mu/'):
    #     os.makedirs('../small_results/VAE_transformer_new/Index/History/mu/')
    # if not os.path.exists('../small_results/VAE_transformer_new/Index/History/sigma/'):
    #     os.makedirs('../small_results/VAE_transformer_new/Index/History/sigma/')
    # for i in range(2, 9):
    #     for j in range(24):
    #         FILE = '{}_{}.npy'.format(i, j)
    #         if os.path.exists(trainFilePath+FILE):
    #             muFILE = '../small_results/VAE_transformer_new/Index/History/mu/mu_{}_{}.csv'.format(i, j)
    #             sigmaFILE = '../small_results/VAE_transformer_new/Index/History/sigma/sigma_{}_{}.csv'.format(i, j)
    #             x = constructSingleData(trainFilePath, FILE)
    #             if x.shape[0] > 0:
    #                 predict_data = np.array(x)
    #                 b = np.isnan(predict_data) # check if there is nan in the data
    #                 if True in b:
    #                     print(i,'_', j, " has nan.")
    #                 predict_dataset = torch.utils.data.TensorDataset(torch.from_numpy(predict_data).float())
    #                 predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=1)
    #                 model = torch.load(save_model)
    #                 model.eval()
    #                 result_mu = []
    #                 result_sigma = []
    #                 for _, x in enumerate(predict_loader):
    #                     x = x[0].to(device)
    #                     _, mu, logvar = model(x)
    #                     result_mu.append(mu.cpu().detach().numpy())
    #                     result_sigma.append(logvar.cpu().detach().numpy())
    #                 result_mu = np.concatenate(result_mu, axis = 0)
    #                 result_sigma = np.concatenate(result_sigma, axis = 0)
    #                 parameteroutput(result_mu, muFILE)
    #                 parameteroutput(result_sigma, sigmaFILE)

    # encoding query
    if not os.path.exists('../small_results/VAE_transformer_new/Index/Query/mu/'):
        os.makedirs('../small_results/VAE_transformer_new/Index/Query/mu/')
    if not os.path.exists('../small_results/VAE_transformer_new/Index/Query/sigma/'):
        os.makedirs('../small_results/VAE_transformer_new/Index/Query/sigma/')
    for i in range(2, 9):
        for j in range(24):
            FILE = '{}_{}.npy'.format(i, j)
            if os.path.exists(trainFilePath+FILE):
                muFILE = '../small_results/VAE_transformer_new/Index/Query/mu/mu_{}_{}.csv'.format(i, j)
                sigmaFILE = '../small_results/VAE_transformer_new/Index/Query/sigma/sigma_{}_{}.csv'.format(i, j)
                x = constructSingleData(queryFilePath, FILE)
                if x.shape[0] > 0:
                    predict_data = np.array(x)
                    b = np.isnan(predict_data) # check if there is nan in the data
                    if True in b:
                        print(i,'_', j, " has nan.")
                    predict_dataset = torch.utils.data.TensorDataset(torch.from_numpy(predict_data).float())
                    predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=1)
                    model = torch.load(save_model)
                    model.eval()
                    result_mu = []
                    result_sigma = []
                    for _, x in enumerate(predict_loader):
                        x = x[0].to(device)
                        _, mu, logvar = model(x)
                        result_mu.append(mu.cpu().detach().numpy())
                        result_sigma.append(logvar.cpu().detach().numpy())
                    result_mu = np.concatenate(result_mu, axis = 0)
                    result_sigma = np.concatenate(result_sigma, axis = 0)
                    # parameteroutput(result_mu, muFILE)
                    # parameteroutput(result_sigma, sigmaFILE)



if __name__ == '__main__':
    main()

