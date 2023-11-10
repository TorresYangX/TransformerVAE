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
linear_dim = 8
output_dim = 2
hidden_dim = 8
latent_dim = 4
MAX_EPOCH = 1000

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


class encoder(nn.Module):

    def __init__(self, hidden_dim, input_dim, output_dim, latent_dim, batch_size):
        super(encoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size

        self.lstm = nn.LSTM(self.output_dim, self.hidden_dim, bias=True, batch_first=True, bidirectional=True)
        self.linear_in = nn.Linear(self.input_dim, self.output_dim, bias=True)
        self.linear_out = nn.Linear(self.output_dim, self.output_dim, bias=True)
        self.mu = nn.Linear(self.hidden_dim * 2, self.latent_dim, bias=True)
        self.log_sigma = nn.Linear(self.hidden_dim * 2, self.latent_dim, bias=True)
        self.relu = nn.ReLU()
        self.softsign = nn.Softsign()
        # self.softmax = nn.Softmax(dim=2)

        self.hidden_init = (
        torch.autograd.Variable(torch.zeros(1 * 2, self.batch_size, self.hidden_dim).to(torch.device('cuda'))),
        torch.autograd.Variable(torch.zeros(1 * 2, self.batch_size, self.hidden_dim)).to(torch.device('cuda')))

    def attention(self, lstm_out, fin_state):
        # lstm_out: torch.Size([100, 20, 512])
        # fin_state: torch.Size([2, 100, 256])
        hidden = fin_state.view(-1, self.hidden_dim * 2, 1)
        # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        # torch.Size([100, 512, 1])
        att_weight = torch.bmm(lstm_out, hidden).squeeze(2)  # 矩阵相乘 去掉维度为1的维度，在位置2
        # attn_weights : [batch_size, n_step]
        # torch.Size([100, 20])
        soft_attn_weights = F.softmax(att_weight, 1)
        # torch.Size([100, 20])  unsqueeze扩张维度，在位置2
        content = torch.bmm(lstm_out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        # context : [batch_size, n_hidden * num_directions(=2)] torch.Size([100, 512])
        return content

    def forward(self, trajectory):
        locrep = self.linear_in(trajectory).cuda()  # trajectory torch.Size([1000, 6, 1])
        locrep = self.softsign(locrep).cuda()
        # locrep = locrep.reshape(900,1,200)
        lstm1, (fin_hidden_state,fin_cell_state) = self.lstm(locrep, self.hidden_init)  # .to(torch.device('cuda'))
        #torch.Size([50, 6, 512])
        #print()
        att_out = self.attention(lstm1, fin_hidden_state)
        # att_out torch.Size([50, 512])
        #lstm2 = lstm1[:, -1, :].to(torch.device('cuda'))
        # lstm2 torch.Size([50, 512])
        mu_ = self.mu(att_out)
        mu_ = self.softsign(mu_)
        sigma_ = self.log_sigma(att_out)
        sigma_ = self.softsign(sigma_)
        return mu_, sigma_


class decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, latent_dim, batch_size, repeat_times):
        super(decoder, self).__init__()
        self.rep = repeat_times
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size

        self.lstm = nn.LSTM(self.latent_dim, self.hidden_dim, batch_first=True, bias=True, bidirectional=True)
        # self.linear = nn.Linear(self.hidden_dim*self.rep, self.hidden_dim, bias=True)#
        self.linear = nn.Linear(self.hidden_dim * 2, self.output_dim, bias=True)
        self.softsign = nn.Softsign()
        self.softmax = nn.Softmax(dim=2)

        self.hidden_init = (
        torch.autograd.Variable(torch.zeros(1 * 2, self.batch_size, self.hidden_dim).to(torch.device('cuda'))),
        torch.autograd.Variable(torch.zeros(1 * 2, self.batch_size, self.hidden_dim)).to(torch.device('cuda')))

    def forward(self, gaussian_noise):
        repvec = gaussian_noise.unsqueeze(1).repeat(1, self.rep, 1).cuda()
        # torch.Size([900, 20, 6])
        lstm1, _ = self.lstm(repvec, self.hidden_init)
        # torch.Size([900, 20, 200])  torch.Size([900, 3, 200])
        # lin = lstm1.contiguous().view(self.batch_size,-1)#torch.Size([900, 600])
        # 540000
        output = self.linear(lstm1).cuda()
        output = self.softmax(output).cuda()
        # output2 = self.linear2(output)
        return output


class GaussianNoise():

    def __init__(self):
        #        super(GaussianNoise, self).__init__()
        self.mean_ = 0.
        self.std_ = 1.

    def sampling(self, mu_, sigma_):
        epsilon = torch.empty(mu_.shape[0], mu_.shape[1]).normal_(mean=self.mean_, std=self.std_).cuda()
        return mu_ + torch.exp(sigma_ / 2) * epsilon
        # sigma torch.Size([1000, 6, 20])
        # epsilon torch.Size([1000, 6])

    def weighted_sampling(self, mu_, sigma_, weight_):
        index_ = list(torch.utils.data.WeightedRandomSampler(weight_.view(-1), int(weight_.sum()), replacement=True))
        mu_all = mu_[index_]
        sigma_all = sigma_[index_]
        epsilon_all = torch.empty(mu_all.shape[0], mu_all.shape[1]).normal_(mean=self.mean_, std=self.std_)
        return index_, mu_all + torch.exp(sigma_all / 2) * epsilon_all


class VAE(nn.Module):
    def __init__(self, input_dim, linear_dim, output_dim, hidden_dim, latent_dim, batch_size, repeat_times):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.linear_dim = linear_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.rep = repeat_times

        self.encoder_ = encoder(self.hidden_dim, self.input_dim, self.linear_dim, self.latent_dim,
                                self.batch_size).cuda()
        self.decoder_ = decoder(self.hidden_dim, self.output_dim, self.latent_dim, self.batch_size, self.rep).cuda()
        self.gaussian_noise = GaussianNoise()

    def loss_fn(self, x_hat, x, mu, logvar):
        BCE = torch.mean((x_hat - x) ** 2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + 0. * KLD

    def forward(self, inputs):
        mu_, sigma_ = self.encoder_(inputs)
        random_noise = self.gaussian_noise.sampling(mu_.cuda(), sigma_.cuda())
        outputs = self.decoder_(random_noise)
        return outputs, mu_, sigma_



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
    save_model = '../small_results/VAE_attention/VAE_attention.pt'
    trainlog = '../small_results/VAE_attention/trainlog.csv'

    trainFilePath = '../small_data/trainingData/'
    queryFilePath = '../small_data/queryData/'
    file = '2_17.npy'

    # # train
    # x, _ = constructTrainingData(trainFilePath, file, BATCH_SIZE)

    # x = (x - (x.max() + x.min()) / 2) / (x.max() - x.min()) * 2

    # train_data = np.array(x)
    # test_data = np.array(x)

    # train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data).float())
    # test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data).float())

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # model = VAE(input_dim, linear_dim, output_dim, hidden_dim, latent_dim, BATCH_SIZE, 60).to(device)
    # optimizer = optim.RMSprop(model.parameters(),lr=0.0005)

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
    # plt.savefig('../small_results/VAE_attention/loss_figure.png')
    # plt.show()


    # # save model
    # torch.save(model, save_model)


    # encoding history
    if not os.path.exists('../small_results/VAE_attention/Index/History/mu/'):
        os.makedirs('../small_results/VAE_attention/Index/History/mu/')
    if not os.path.exists('../small_results/VAE_attention/Index/History/sigma/'):
        os.makedirs('../small_results/VAE_attention/Index/History/sigma/')
    for i in range(2, 9):
        for j in range(24):
            FILE = '{}_{}.npy'.format(i, j)
            if os.path.exists(trainFilePath+FILE):
                muFILE = '../small_results/VAE_attention/Index/History/mu/mu_{}_{}.csv'.format(i, j)
                sigmaFILE = '../small_results/VAE_attention/Index/History/sigma/sigma_{}_{}.csv'.format(i, j)
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
                    result_mu = torch.zeros(16, 4).to(device)
                    result_sigma = torch.zeros(16, 4).to(device)
                    for _, x in enumerate(predict_loader):
                        x = x[0].to(device)
                        _, mu, logvar = model(x)
                        result_mu=torch.cat((result_mu, mu), 0)
                        result_sigma=torch.cat((result_sigma, logvar), 0)
                    parameteroutput(result_mu[16:,:].cpu().detach(), muFILE)
                    parameteroutput(result_sigma[16:,:].cpu().detach(), sigmaFILE)

    # encoding query
    if not os.path.exists('../small_results/VAE_attention/Index/Query/mu/'):
        os.makedirs('../small_results/VAE_attention/Index/Query/mu/')
    if not os.path.exists('../small_results/VAE_attention/Index/Query/sigma/'):
        os.makedirs('../small_results/VAE_attention/Index/Query/sigma/')
    for i in range(2, 9):
        for j in range(24):
            FILE = '{}_{}.npy'.format(i, j)
            if os.path.exists(trainFilePath+FILE):
                muFILE = '../small_results/VAE_attention/Index/Query/mu/mu_{}_{}.csv'.format(i, j)
                sigmaFILE = '../small_results/VAE_attention/Index/Query/sigma/sigma_{}_{}.csv'.format(i, j)
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
                    result_mu = torch.zeros(16, 4).to(device)
                    result_sigma = torch.zeros(16, 4).to(device)
                    for _, x in enumerate(predict_loader):
                        x = x[0].to(device)
                        _, mu, logvar = model(x)
                        result_mu=torch.cat((result_mu, mu), 0)
                        result_sigma=torch.cat((result_sigma, logvar), 0)
                    parameteroutput(result_mu[16:,:].cpu().detach(), muFILE)
                    parameteroutput(result_sigma[16:,:].cpu().detach(), sigmaFILE)


if __name__ == '__main__':
    main()