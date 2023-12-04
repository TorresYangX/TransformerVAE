import torch
import torch.nn as nn
import torch.optim as optim
from nvib.denoising_attention import DenoisingMultiheadAttention
from nvib.kl import kl_dirichlet, kl_gaussian
from nvib.nvib_layer import Nvib

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import logging
import os
from tqdm import trange
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

grid_num = 50
vocab_size = grid_num * grid_num +2 #V
Batch_size = 16 #B
embedding_dim = 16 # H
PRIOR_MU = 0
PRIOR_VAR = 1
PRIOR_ALPHA = 1
KAPPA = 1
DELTA = 1
KL_GAUSSIAN_LAMBDA = 0.001
KL_DIRICHLET_LAMBDA = 1
KL_ANNEALING_GAUSSIAN = "constant"
KL_ANNEALING_DIRICHLET = "constant"


dropout = 0.1
learning_rate = 0.001
MAX_EPOCH = 20
ACCUMULATION_STEPS = 1

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



def kl_annealing(start=0, stop=1, n_epoch=30, type="constant", n_cycle=4, ratio=0.5):
    """
    Cyclic and monotonic cosine KL annealing from:
    https://github.com/haofuml/cyclical_annealing/blob/6ef4ebabb631df696cf4bfc333a965283eba1958/plot/plot_schedules.ipynb

    :param start:0
    :param stop:1
    :param n_epoch:Total epochs
    :param type: Type of annealing "constant", "monotonic" or "cyclic"
    :param n_cycle:
    :param ratio:
    :return: a list of all factors
    """
    L = np.ones(n_epoch)
    if type != "constant":
        if type == "monotonic":
            n_cycle = 1
            ratio = 0.25

        period = n_epoch / n_cycle
        step = (stop - start) / (period * ratio)

        for c in range(n_cycle):

            v, i = start, 0
            while v <= stop:
                L[int(i + c * period)] = 0.5 - 0.5 * math.cos(v * math.pi)
                v += step
                i += 1
    return L

KL_ANNEALING_FACTOR_GAUSSIAN_LIST = kl_annealing(
    n_epoch=MAX_EPOCH, type=KL_ANNEALING_GAUSSIAN
)
KL_ANNEALING_FACTOR_DIRICHLET_LIST = kl_annealing(
    n_epoch=MAX_EPOCH, type=KL_ANNEALING_DIRICHLET
)


class TokenEmbedding(nn.Module):
    def __init__(self):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

    def forward(self, x):
        return self.embedding(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=1000, mul_by_sqrt=True):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        self.mul_by_sqrt = mul_by_sqrt

    def forward(self, x):
        x = x.permute(1, 0, 2)
        if self.mul_by_sqrt:
            x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        pe = self.pe[:, 1 : seq_len + 1]
        pe = pe.expand_as(x)
        x = x + pe
        x = x.permute(1, 0, 2)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=embedding_dim, 
                                                                  nhead=8, 
                                                                  dim_feedforward=2048, 
                                                                  dropout=0.1, 
                                                                  activation='relu', 
                                                                  batch_first=False, 
                                                                  device=device) 
        self.TransformerEncoder = nn.TransformerEncoder(self.TransformerEncoderLayer, num_layers=6)
        
    def forward(self, src, src_key_padding_mask):
        return self.TransformerEncoder(src, src_key_padding_mask = src_key_padding_mask)
    
class TransformerDecoder(nn.Module):
    def __init__(self):
        super(TransformerDecoder, self).__init__()
        self.TransformerDecoderLayer = nn.TransformerDecoderLayer(d_model=embedding_dim, 
                                                                  nhead=1, 
                                                                  dim_feedforward=2048, 
                                                                  dropout=0.1, 
                                                                  activation='relu', 
                                                                  batch_first=False, 
                                                                  device=device) 
        self.TransformerDecoder = nn.TransformerDecoder(self.TransformerDecoderLayer, num_layers=1)
        for layer_num, layer in enumerate(self.TransformerDecoder.layers):
            layer.multihead_attn = DenoisingMultiheadAttention(embed_dim=embedding_dim,
                                                        num_heads=1,
                                                        dropout=0.1,
                                                        bias=False)

    def forward(self, tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
        return self.TransformerDecoder(tgt = tgt, 
                                       memory=memory, 
                                       tgt_mask=tgt_mask, 
                                       tgt_key_padding_mask=tgt_key_padding_mask, 
                                       memory_key_padding_mask=memory_key_padding_mask)
    
    

class TransformerNvib(nn.Module):
    '''
    Data format:
    SRC: ... [EOS]
    TGT: ... [EOS]
    '''
    def __init__(self):
        super(TransformerNvib, self).__init__()
        self.token_embedding = TokenEmbedding()
        self.position_embedding = PositionalEncoding(embedding_dim)
        self.transformer_encoder = TransformerEncoder()
        self.nvib = Nvib(
            size_in = embedding_dim,
            size_out = embedding_dim,
            prior_mu = PRIOR_MU,
            prior_var = PRIOR_VAR,
            prior_alpha = PRIOR_ALPHA,
            kappa = KAPPA,
            delta = DELTA,
        )
        self.transformer_decoder = TransformerDecoder()
        self.output_proj = nn.Linear(embedding_dim, vocab_size)
        self.drop = nn.Dropout(dropout)

    def encode(self,src, src_key_padding_mask):
        src = self.token_embedding(src.to(torch.int64).to(device)) #(trajectory_length, Batch_size, embedding_dim) (60,64,512)
        src = self.drop(src)
        src = self.position_embedding(src).to(torch.float32) #(trajectory_length, Batch_size, embedding_dim) (60,64,512)
        src = self.transformer_encoder(src, src_key_padding_mask)
        return src
    
    def decode(self, tgt, memory, memory_key_padding_mask, tgt_key_padding_mask):
        tgt = self.token_embedding(tgt.to(torch.int64).to(device))
        tgt = self.drop(tgt)
        tgt = self.position_embedding(tgt).to(torch.float32) 
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[0]).to(device) 
        output = self.transformer_decoder(
            tgt=tgt,  # [Nt,B,H] (trajectory_length+1, Batch_size, embedding_dim) (61,64,512)
            memory=memory,  # [Nt,B,H] (trajectory_length+1, Batch_size, embedding_dim) (61,64,512)
            tgt_mask=tgt_mask,  # [Nt,Nt] (61,61)
            tgt_key_padding_mask=tgt_key_padding_mask,  # [B,Nt] (64,61)
            memory_key_padding_mask=memory_key_padding_mask, # [B,Nt] (64,61)
        )
        logits = self.output_proj(output)  # [Nt,B,V]
        return logits

    def loss(self, logits, targets, epoch,  **kwargs):
        """
        Calculate the loss

        :param logits: output of the decoder [Nt,B,V]
        :param targets: target token ids [B,Nt]
        :return: Dictionary of scalar valued losses. With a value "Loss" to backprop
        """

        # KL loss averaged over batches

        kl_loss_g = torch.mean(
            kl_gaussian(
                prior_mu=PRIOR_MU,
                prior_var=PRIOR_VAR,
                kappa=KAPPA,
                **kwargs
            )
        )
        kl_loss_d = torch.mean(
            kl_dirichlet(
                prior_alpha=PRIOR_ALPHA,
                delta=DELTA,
                kappa=KAPPA,
                **kwargs
            )
        )

        # Cross Entropy where pad = 0
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        # Transform targets
        targets = torch.flatten(targets)  # [Nt x B]
        # Transform vocabulary
        logits = torch.flatten(logits, start_dim=0, end_dim=1)  # [Nt x B, V]
        # Calculates loss over [Nt x B]
        cross_entropy_loss = criterion(logits.float(), targets.long())  # [Nt x B]
        # Average loss + average KL for backprop and sum loss for logging
        KL_ANNEALING_FACTOR_GAUSSIAN = KL_ANNEALING_FACTOR_GAUSSIAN_LIST[epoch-1] 
        KL_ANNEALING_FACTOR_DIRICHLET = KL_ANNEALING_FACTOR_DIRICHLET_LIST[epoch-1]
        return {
            "Loss": torch.mean(cross_entropy_loss)
            + KL_GAUSSIAN_LAMBDA * KL_ANNEALING_FACTOR_GAUSSIAN * kl_loss_g
            + KL_DIRICHLET_LAMBDA * KL_ANNEALING_FACTOR_DIRICHLET * kl_loss_d,
            "CrossEntropy": torch.sum(cross_entropy_loss),
            "KLGaussian": kl_loss_g,
            "KLDirichlet": kl_loss_d,
        }

    def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask):
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask) #(trajectory_length, Batch_size, embedding_dim) (60,64,512)
        latent_output_dict = self.nvib(memory, src_key_padding_mask)
        output = self.decode(
            tgt=tgt,
            # latent_output_dict["z"]: tuple(z, pi, mu, logvar), 
            # z:(trajectory_length+1, Batch_size, embedding_dim), 
            # pi:(trajectory_length+1, Batch_size, 1), ##nan##
            # mu:(trajectory_length+1, Batch_size, embedding_dim),
            # logvar:(trajectory_length+1, Batch_size, embedding_dim),
            memory=latent_output_dict["z"], 
            tgt_key_padding_mask=tgt_key_padding_mask,  # [B,Nt]
            memory_key_padding_mask=latent_output_dict["memory_key_padding_mask"],
        )  # [B,Nl] 
        return {
            "logits": output,  # [Nt, B, V]
            **latent_output_dict,
        }

def training(model, train_loader, OPTIMIZER, trajectory_length, epoch):
    train_losses_value = 0
    for idx, x in enumerate(train_loader):
        x = x[0].transpose(0,1).to(device)
        eos = torch.full((1, x.shape[1]), grid_num*grid_num).to(device)
        sos = torch.full((1, x.shape[1]), grid_num*grid_num+1).to(device)
        src = torch.cat([x, eos], dim=0)
        tgt = torch.cat([sos, x], dim=0)

        src_key_padding_mask = torch.zeros((x.shape[1], trajectory_length + 1), dtype=torch.bool).to(device)
        tgt_key_padding_mask = torch.zeros((x.shape[1], trajectory_length + 1), dtype=torch.bool).to(device)

        train_outputs_dict = model(
            src,
            tgt,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # [sq_len, bs, Vocab]
        # print(train_outputs_dict["logits"].shape) (61,64,2501)
        train_losses = model.loss(**train_outputs_dict, targets=tgt, epoch=epoch)
        (train_losses["Loss"] / ACCUMULATION_STEPS).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        if ((idx + 1) % ACCUMULATION_STEPS == 0) or (idx + 1 == len(train_loader)):
            OPTIMIZER.step()
            OPTIMIZER.zero_grad()
        train_losses_value += train_losses["Loss"].item()
    return train_losses_value / len(train_loader.dataset)

def evaluation(model, test_loader, trajectory_length, epoch):
    test_losses_value = 0
    for idx, x in enumerate(test_loader):
        with torch.no_grad():
            x = x[0].transpose(0,1).to(device)
            eos = torch.full((1, x.shape[1]), grid_num*grid_num).to(device)
            sos = torch.full((1, x.shape[1]), grid_num*grid_num+1).to(device)
            src = torch.cat([x, eos], dim=0)
            tgt = torch.cat([sos, x], dim=0)

            src_key_padding_mask = torch.zeros((x.shape[1], trajectory_length + 1), dtype=torch.bool).to(device)
            tgt_key_padding_mask = torch.zeros((x.shape[1], trajectory_length + 1), dtype=torch.bool).to(device)

            test_outputs_dict = model(
                src,
                tgt,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )  # [sq_len, bs, Vocab]
            test_losses = model.loss(**test_outputs_dict, targets=tgt, epoch=epoch)
            test_losses_value += test_losses["Loss"].item()
    return test_losses_value / len(test_loader.dataset)


def trainModel(trainFilePath, modelSavePath, trainlogPath, trajectory_length, isSSM):
    x = constructTrainingData(trainFilePath, Batch_size)
    dataSet = torch.utils.data.TensorDataset(torch.from_numpy(x))
    val_size = int(0.2 * len(dataSet))

    model = TransformerNvib().to(device)

    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    logger = get_logger(trainlogPath)
    train_loss_list = []
    test_loss_list = []
    print("Start training...")
    for epoch in trange(MAX_EPOCH):
        train_dataset, test_dataset = torch.utils.data.random_split(dataSet, [len(dataSet) - val_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Batch_size)
        
        train_loss = training(model, train_loader, optimizer, trajectory_length, epoch)
        test_loss = evaluation(model, test_loader, trajectory_length, epoch)
        logger.info('Epoch:[{}/{}]\t Train Loss={:.4f}\t Test Loss={:.4f}'.format(epoch+1 , MAX_EPOCH, train_loss, test_loss ))
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
    print("End training...")
    torch.save(model, modelSavePath)
    plot_loss(train_loss_list, test_loss_list, isSSM)


def encoding(modelPath, dataPath, trajectory_length, isSSM):
    if not isSSM:
        muFolder = '../results/VAE_nvib/Index/mu/'
        sigmaFolder = '../results/VAE_nvib/Index/sigma/'
        piFolder = '../results/VAE_nvib/Index/pi/'
        alphaFolder = '../results/VAE_nvib/Index/alpha/'
    else:
        dbNUM = dataPath.split('/')[-3]
        muFolder = '../SSM_KNN/VAE_nvib/Index/{}/mu/'.format(dbNUM)
        sigmaFolder = '../SSM_KNN/VAE_nvib/Index/{}/sigma/'.format(dbNUM)
        piFolder = '../SSM_KNN/VAE_nvib/Index/{}/pi/'.format(dbNUM)
        alphaFolder = '../SSM_KNN/VAE_nvib/Index/{}/alpha/'.format(dbNUM)
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
                x = constructSingleData(dataPath, FILE, Batch_size)
                if x.shape[0] > 0:
                    predict_data = np.array(x)
                    predict_dataset = torch.utils.data.TensorDataset(torch.from_numpy(predict_data))
                    predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=Batch_size)
                    model = torch.load(modelPath)
                    model.eval()
                    result_mu = []
                    result_sigma = []
                    result_pi = []
                    result_alpha = []
                    for idx, x in enumerate(predict_loader):
                        x = x[0].transpose(0,1).to(device)
                        eos = torch.full((1, x.shape[1]), grid_num*grid_num).to(device)
                        sos = torch.full((1, x.shape[1]), grid_num*grid_num+1).to(device)
                        src = torch.cat([x, eos], dim=0)
                        tgt = torch.cat([sos, x], dim=0)    

                        src_key_padding_mask = torch.zeros((x.shape[1], trajectory_length + 1), dtype=torch.bool).to(device)
                        tgt_key_padding_mask = torch.zeros((x.shape[1], trajectory_length + 1), dtype=torch.bool).to(device)

                        # Forward pass
                        outputs_dict = model(
                            src,
                            tgt,
                            src_key_padding_mask=src_key_padding_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                        ) 
                        mu = outputs_dict["mu"].mean(dim=0, keepdim=True) #(1,16,16)
                        logvar = outputs_dict["logvar"].mean(dim=0, keepdim=True) #(1,16,16)
                        pi = outputs_dict["pi"].repeat(1,1,embedding_dim) #(61,16,16)
                        alpha = outputs_dict["alpha"].repeat(1,1,embedding_dim) #(61,16,16)
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


def main(args):
    if not args.SSM_KNN:
        trajectory_length = 60
        root = '../results/VAE_nvib/'
        if not os.path.exists(root):
            os.makedirs(root)
        save_model = '../results/VAE_nvib/VAE_nvib.pt'
        trainlog = '../results/VAE_nvib/trainlog.csv'
        trainFilePath = '../data/Train/trainGridData/'
        dataPath = '../data/Experiment/experimentGridData/'
        if args.model=="train":
            trainModel(trainFilePath, save_model, trainlog, trajectory_length, args.SSM_KNN)
        else:
            encoding(save_model, dataPath, trajectory_length, args.SSM_KNN)
    else:
        trajectory_length = 30
        root = '../SSM_KNN/'
        if not os.path.exists(root):
            os.makedirs(root)
        root = root + 'VAE_nvib/'
        if not os.path.exists(root):
            os.makedirs(root)
        save_model = '../SSM_KNN/VAE_nvib/VAE_nvib.pt'
        trainlog = '../SSM_KNN/VAE_nvib/trainlog.csv'
        trainFilePath = '../data/Train/SSM_KNN/DataBase/GridData/'
        dataPath_1 = '../data/Experiment/SSM_KNN/DataBase_1/GridData/'
        dataPath_2 = '../data/Experiment/SSM_KNN/DataBase_2/GridData/'
        if args.model=="train":
            trainModel(trainFilePath, save_model, trainlog, trajectory_length, args.SSM_KNN)
        else:
            encoding(save_model, dataPath_1, trajectory_length, args.SSM_KNN)
            encoding(save_model, dataPath_2, trajectory_length, args.SSM_KNN)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", type=str, default="train", choices=["train","encode"] ,help="train or encode", required=True)

    parser.add_argument("-s", "--SSM_KNN", type=bool, default=False)

    args = parser.parse_args()

    main(args)

