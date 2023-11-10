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

reconpath = '../results/AE/'

# same
def constructTrainingData(filePath, file, BATCH_SIZE):
    data = np.load(filePath + file)
    resid = (data.shape[0] // BATCH_SIZE) * BATCH_SIZE
    data = data[:resid, :, :]
    data[:, :, 0] = (116.4 - data[:, :, 0]) / 0.4
    data[:, :, 1] = (39.9 - data[:, :, 1]) / 0.3
    return data[:, :,:2], data[:, :, 3]


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


class AE(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 latent_dim,
                 batch_size=16,
                 seq_len=60,
                 ):
        super(AE, self).__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.latent_size = latent_dim
        self.batch_size = batch_size
        self.seq_len = seq_len


        self.encoder_lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.encoder = nn.Linear(self.hidden_size, self.latent_size)

        self.decoder_lstm = nn.LSTM(input_size=self.latent_size, hidden_size=self.hidden_size, batch_first=True)
        self.decoder_fc = nn.Linear(self.hidden_size, self.input_size)

        self.hidden_init = (torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size).to(device)),
                            torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size)).to(device))

        self.softsign = nn.Softsign()

    def encode(self, x):
        h, _ = self.encoder_lstm(x, self.hidden_init)
        h = h[:, -1:, :]
        h = self.softsign(self.encoder(h))
        return h

    def decode(self, h):
        h = h.repeat(1, self.seq_len, 1)
        h, _ = self.decoder_lstm(h, self.hidden_init)
        x_hat = self.softsign(self.decoder_fc(h))
        return x_hat

    def forward(self, x):
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h

def train(model, train_loader, optimizer):
    model.train()
    train_loss = 0
    for _, x in enumerate(train_loader):
        x = x[0].to(device)
        optimizer.zero_grad()
        x_hat, _ = model(x)
        loss = F.mse_loss(x_hat, x)
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
            x_hat, _ = model(x)
            test_loss += F.mse_loss(x_hat, x).item()
    return test_loss / len(test_loader.dataset)


def main():
    # define path
    folder_path = '../small_results/AE/'
    save_model = '../small_results/AE/AE.pt'
    trainlog = '../small_results/AE/trainlog.csv'
    aeweight = '../small_results/AE/ae.h5'
    decoder_lstm_weight = '../small_results/AE/decoder_lstm.h5'
    decoder_fc_weight = '../small_results/AE/decoder_fc.h5'

    trainFilePath = '../small_data/trainingData/'
    queryFilePath = '../small_data/queryData/'
    file = '2_17.npy'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # train
    x, _ = constructTrainingData(trainFilePath, file, BATCH_SIZE)

    x = (x - (x.max() + x.min()) / 2) / (x.max() - x.min()) * 2

    train_data = np.array(x)
    test_data = np.array(x)

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data).float())
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data).float())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = AE(input_dim,hidden_dim,latent_dim).to(device)
    optimizer = optim.RMSprop(model.parameters(),lr=0.0001)

    logger = get_logger(trainlog)
    train_loss_list = []
    test_loss_list = []
    for epoch in range(MAX_EPOCH):
        train_loss = train(model, train_loader, optimizer)
        test_loss = test(model, test_loader)
        logger.info('Epoch:[{}/{}]\t Train Loss={:.4f}\t Test Loss={:.4f}'.format(epoch+1 , MAX_EPOCH, train_loss, test_loss ))
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
    x=[i for i in range(len(train_loss_list))]
    figure = plt.figure(figsize=(20, 8), dpi=80)
    plt.plot(x,train_loss_list,label='train_losses')
    plt.plot(x,test_loss_list,label='test_losses')
    plt.xlabel("iterations",fontsize=15)
    plt.ylabel("loss",fontsize=15)
    plt.legend()
    plt.grid()
    plt.savefig('../small_results/AE/loss_figure.png')
    plt.show()


    # save model
    torch.save(model, save_model)
    torch.save(model.state_dict(), aeweight)
    torch.save(model.decoder_lstm.state_dict(), decoder_lstm_weight)
    torch.save(model.decoder_fc.state_dict(), decoder_fc_weight)


    # encoding history
    if not os.path.exists('../small_results/AE/Index/History/'):
        os.makedirs('../small_results/AE/Index/History/')
    for i in range(2, 9):
        for j in range(24):
            FILE = '{}_{}.npy'.format(i, j)
            if os.path.exists(trainFilePath+FILE):
                probFILE = '../small_results/AE/Index/History/prob_{}_{}.csv'.format(i, j)
                x, _ = constructTrainingData(trainFilePath, FILE, BATCH_SIZE)
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
                    result_prob = torch.zeros(16, 1, 4).to(device)
                    for _, x in enumerate(predict_loader):
                        x = x[0].to(device)
                        _, h = model(x)
                        result_prob=torch.cat((result_prob, h), 0)
                    parameteroutput(result_prob[16:,0,:].cpu().detach(), probFILE)

    # encoding query
    if not os.path.exists('../small_results/AE/Index/Query/'):
        os.makedirs('../small_results/AE/Index/Query/')
    for i in range(2, 9):
        for j in range(24):
            FILE = '{}_{}.npy'.format(i, j)
            if os.path.exists(trainFilePath+FILE):
                probFILE = '../small_results/AE/Index/Query/prob_{}_{}.csv'.format(i, j)
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
                    result_prob = torch.zeros(16, 1, 4).to(device)
                    for _, x in enumerate(predict_loader):
                        x = x[0].to(device)
                        _, h = model(x)
                        result_prob=torch.cat((result_prob, h), 0)
                    parameteroutput(result_prob[16:,0,:].cpu().detach(), probFILE)


if __name__ == '__main__':
    main()
