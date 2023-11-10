import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
input_dim = 2
hidden_dim = 8
latent_dim = 4
MAX_EPOCH = 2000

reconpath = '../results/VariationalAE/'

# same
def constructTrainingData(filePath, file, BATCH_SIZE):
    data = np.load(filePath + file)
    resid = (data.shape[0] // BATCH_SIZE) * BATCH_SIZE
    data = data[:resid, :, :]
    data[:, :, 0] = (116.4 - data[:, :, 0]) / 0.4
    data[:, :, 1] = (39.9 - data[:, :, 1]) / 0.3
    return data[:, :,:2], data[:, :, 3] ##only return lon and lat

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


class VAE(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 latent_dim,
                 batch_size=16,
                 seq_len=60,
                 ):
        super(VAE, self).__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.latent_size = latent_dim
        self.batch_size = batch_size
        self.seq_len = seq_len


        self.encoder_lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.encoder_mu = nn.Linear(self.hidden_size, self.latent_size)
        self.encoder_logvar = nn.Linear(self.hidden_size, self.latent_size)

        self.decoder_lstm = nn.LSTM(input_size=self.latent_size, hidden_size=self.hidden_size, batch_first=True)
        self.decoder_fc = nn.Linear(self.hidden_size, self.input_size)

        self.hidden_init = (torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size).to(device)),
                            torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size)).to(device))

        self.softsign = nn.Softsign()

    def encode(self, x):
        h, _ = self.encoder_lstm(x, self.hidden_init)
        h = h[:, -1:, :]
        mu = self.softsign(self.encoder_mu(h))
        logvar = self.softsign(self.encoder_logvar(h))
        return mu, logvar

    # 对应sampling
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        z = z.repeat(1, self.seq_len, 1)
        h, _ = self.decoder_lstm(z, self.hidden_init)
        x_hat = self.softsign(self.decoder_fc(h))
        return x_hat

    def loss_fn(self, x_hat, x, mu, logvar):
        BCE = torch.mean((x_hat - x) ** 2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + 0. * KLD

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

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
    save_model = '../small_results/VariationalAE/VAE.pt'
    trainlog = '../small_results/VariationalAE/trainlog.csv'
    vaeweight = '../small_results/VariationalAE/vae.h5'
    decoder_lstm_weight = '../small_results/VariationalAE/decoder_lstm.h5'
    decoder_fc_weight = '../small_results/VariationalAE/decoder_fc.h5'
    encoder_mu_weight = '../small_results/VariationalAE/encoder_mu.h5'
    encoder_sigma_weight = '../small_results/VariationalAE/encoder_sigma.h5'

    trainFilePath = '../small_data/trainingData/'
    queryFilePath = '../small_data/queryData/'
    file = '2_17.npy'

    # # train
    # x, y = constructTrainingData(trainFilePath, file, BATCH_SIZE)

    # x = (x - (x.max() + x.min()) / 2) / (x.max() - x.min()) * 2

    # train_data = np.array(x)
    # test_data = np.array(x)

    # train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data).float())
    # test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data).float())

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # model = VAE(input_dim,hidden_dim,latent_dim).to(device)
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
    # plt.savefig('../small_results/VariationalAE/loss_figure.png')
    # plt.show()


    # # save model
    # torch.save(model, save_model)
    # torch.save(model.state_dict(), vaeweight)
    # torch.save(model.decoder_lstm.state_dict(), decoder_lstm_weight)
    # torch.save(model.decoder_fc.state_dict(), decoder_fc_weight)
    # torch.save(model.encoder_mu.state_dict(), encoder_mu_weight)
    # torch.save(model.encoder_logvar.state_dict(), encoder_sigma_weight)


    # # encoding history
    # if not os.path.exists('../small_results/VariationalAE/Index/History/mu/'):
    #     os.makedirs('../small_results/VariationalAE/Index/History/mu/')
    # if not os.path.exists('../small_results/VariationalAE/Index/History/sigma/'):
    #     os.makedirs('../small_results/VariationalAE/Index/History/sigma/')
    # for i in range(2, 9):
    #     for j in range(24):
    #         FILE = '{}_{}.npy'.format(i, j)
    #         if os.path.exists(trainFilePath+FILE):
    #             muFILE = '../small_results/VariationalAE/Index/History/mu/mu_{}_{}.csv'.format(i, j)
    #             sigmaFILE = '../small_results/VariationalAE/Index/History/sigma/sigma_{}_{}.csv'.format(i, j)
    #             x, _ = constructTrainingData(trainFilePath, FILE, BATCH_SIZE)
    #             if x.shape[0] > 0:
    #                 x = (x - (x.max() + x.min()) / 2) / (x.max() - x.min()) * 2
    #                 predict_data = np.array(x)
    #                 b = np.isnan(predict_data) # check if there is nan in the data
    #                 if True in b:
    #                     print(i,'_', j, " has nan.")
    #                 predict_dataset = torch.utils.data.TensorDataset(torch.from_numpy(predict_data).float())
    #                 predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=BATCH_SIZE)
    #                 model = torch.load(save_model)
    #                 model.eval()
    #                 result_mu = torch.zeros(16, 1, 4).to(device)
    #                 result_sigma = torch.zeros(16, 1, 4).to(device)
    #                 for _, x in enumerate(predict_loader):
    #                     x = x[0].to(device)
    #                     _, mu, logvar = model(x)
    #                     result_mu=torch.cat((result_mu, mu), 0)
    #                     result_sigma=torch.cat((result_sigma, logvar), 0)
    #                 parameteroutput(result_mu[16:,0,:].cpu().detach(), muFILE)
    #                 parameteroutput(result_sigma[16:,0,:].cpu().detach(), sigmaFILE)

    # encoding query
    if not os.path.exists('../small_results/VariationalAE/Index/Query/mu/'):
        os.makedirs('../small_results/VariationalAE/Index/Query/mu/')
    if not os.path.exists('../small_results/VariationalAE/Index/Query/sigma/'):
        os.makedirs('../small_results/VariationalAE/Index/Query/sigma/')
    for i in range(2, 9):
        for j in range(24):
            FILE = '{}_{}.npy'.format(i, j)
            if os.path.exists(trainFilePath+FILE):
                muFILE = '../small_results/VariationalAE/Index/Query/mu/mu_{}_{}.csv'.format(i, j)
                sigmaFILE = '../small_results/VariationalAE/Index/Query/sigma/sigma_{}_{}.csv'.format(i, j)
                x, _ = constructTrainingData(queryFilePath, FILE, BATCH_SIZE)
                if x.shape[0] > 0:
                    x = (x - (x.max() + x.min()) / 2) / (x.max() - x.min()) * 2
                    predict_data = np.array(x)
                    b = np.isnan(predict_data) # check if there is nan in the data
                    if True in b:
                        print(i,'_', j, " has nan.")
                    predict_dataset = torch.utils.data.TensorDataset(torch.from_numpy(predict_data).float())
                    predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=BATCH_SIZE)
                    model = torch.load(save_model)
                    model.eval()
                    result_mu = torch.zeros(16, 1, 4).to(device)
                    result_sigma = torch.zeros(16, 1, 4).to(device)
                    for _, x in enumerate(predict_loader):
                        x = x[0].to(device)
                        _, mu, logvar = model(x)
                        # result_mu=torch.cat((result_mu, mu), 0)
                        # result_sigma=torch.cat((result_sigma, logvar), 0)
                    # parameteroutput(result_mu[16:,0,:].cpu().detach(), muFILE)
                    # parameteroutput(result_sigma[16:,0,:].cpu().detach(), sigmaFILE)


if __name__ == '__main__':
    main()
