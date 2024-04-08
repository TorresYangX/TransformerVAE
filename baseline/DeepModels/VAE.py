import os
import time
import torch
import torch.nn as nn
from utils import tool_funcs
from datetime import datetime
from model_config import ModelConfig
from dataset_config import DatasetConfig
from utils.dataloader import read_traj_dataset
from torch.utils.data.dataloader import DataLoader

import logging
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().setLevel(logging.DEBUG)



class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.embedding_dim = ModelConfig.VAE.embedding_dim
        self.trajectory_length = ModelConfig.VAE.traj_len
        self.hidden_dim = ModelConfig.VAE.hidden_dim
        self.latent_dim = ModelConfig.VAE.latent_dim
        self.vocab_size = ModelConfig.VAE.vocab_size
        self.BATCH_SIZE = ModelConfig.VAE.BATCH_SIZE
        
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        self.encoder_lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.encoder_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.encoder_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        self.decoder_lstm = nn.LSTM(input_size=self.latent_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.decoder_fc = nn.Linear(self.hidden_dim, self.embedding_dim)
        self.outfc = nn.Linear(self.embedding_dim, self.vocab_size)
        self.softmax = nn.Softmax(dim=2)

        self.hidden_init = (torch.autograd.Variable(torch.zeros(1, self.BATCH_SIZE, self.hidden_dim).to(ModelConfig.device)),
                            torch.autograd.Variable(torch.zeros(1, self.BATCH_SIZE, self.hidden_dim).to(ModelConfig.device)))

        self.softsign = nn.Softsign()

    def encode(self, x):
        x = self.embedding(x.to(torch.int64))
        h, _ = self.encoder_lstm(x, self.hidden_init)
        h = h[:, -1:, :]
        mu = self.softsign(self.encoder_mu(h))
        logvar = self.softsign(self.encoder_logvar(h))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        z = z.repeat(1, self.trajectory_length, 1)
        h, _ = self.decoder_lstm(z, self.hidden_init)
        x_hat = self.softsign(self.decoder_fc(h))
        x_hat = self.outfc(x_hat)
        x_hat = self.softmax(x_hat)
        return x_hat

    def loss_fn(self, logits, targets, mu, logvar):
        KL_WEIGHT = 0.01
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        targets = torch.flatten(targets)
        logits = torch.flatten(logits, start_dim=0, end_dim=1)
        BCE = criterion(logits.float(), targets.long()).mean()
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return {
            "Loss": BCE + KL_WEIGHT * KLD,
            "BCE": BCE,
            "KLD": KLD,
        }

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return {
            "logits": x_hat,
            "mu": mu,
            "logvar": logvar,
        }
        
        
class VAE_Trainer:
    def __init__(self):
        super(VAE_Trainer, self).__init__()
        
        self.model = VAE().to(ModelConfig.device)
        self.checkpoint_file = '{}/{}_VAE_best.pt'.format(ModelConfig.VAE.checkpoint_dir, DatasetConfig.dataset)
        
        
    def train(self):
        training_starttime = time.time()
        train_dataset = read_traj_dataset(DatasetConfig.grid_total_file)
        train_dataloader = DataLoader(train_dataset, 
                                    batch_size=ModelConfig.VAE.BATCH_SIZE, 
                                    shuffle=False, 
                                    num_workers=0, 
                                    drop_last=True)
        
        training_gpu_usage = training_ram_usage = 0.0
        logging.info("[Training] START! timestamp={}".format(datetime.fromtimestamp(training_starttime)))
        torch.autograd.set_detect_anomaly(True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr = ModelConfig.VAE.learning_rate, weight_decay = 0.0001)
        
        best_loss_train = 100000
        best_epoch = 0
        bad_counter = 0
        bad_patience = ModelConfig.VAE.training_bad_patience
        
        for i_ep in range(ModelConfig.VAE.MAX_EPOCH):
            _time_ep = time.time()
            loss_ep = []
            train_gpu = []
            train_ram = []
            
            self.model.train()
            
            _time_batch_start = time.time()
            for i_batch, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                
                train_dict = self.model(batch.to(ModelConfig.device))
                train_loss = self.model.loss_fn(targets=batch.to(ModelConfig.device), **train_dict)
                
                train_loss['Loss'].backward()
                optimizer.step()
                optimizer.zero_grad()
                
                loss_ep.append(train_loss['Loss'].item())
                train_gpu.append(tool_funcs.GPUInfo.mem()[0])
                train_ram.append(tool_funcs.RAMInfo.mem())
                
                if i_batch % 100 == 0 and i_batch:
                    logging.debug("[Training] ep-batch={}-{}, loss={:.3f}, @={:.3f}, gpu={}, ram={}" \
                            .format(i_ep, i_batch, train_loss['Loss'].item(), time.time() - _time_batch_start,
                                    tool_funcs.GPUInfo.mem(), tool_funcs.RAMInfo.mem()))
                
            loss_ep_avg = tool_funcs.mean(loss_ep)
            logging.info("[Training] ep={}: avg_loss={:.3f}, @={:.3f}/{:.3f}, gpu={}, ram={}" \
                    .format(i_ep, loss_ep_avg, time.time() - _time_ep, time.time() - training_starttime,
                    tool_funcs.GPUInfo.mem(), tool_funcs.RAMInfo.mem()))
            
            training_gpu_usage = tool_funcs.mean(train_gpu)
            training_ram_usage = tool_funcs.mean(train_ram)
            
            # early stopping
            if loss_ep_avg < best_loss_train:
                best_epoch = i_ep
                best_loss_train = loss_ep_avg
                bad_counter = 0
                self.save_checkpoint()
            else:
                bad_counter += 1
                
            if bad_counter == bad_patience or (i_ep + 1) == ModelConfig.VAE.MAX_EPOCH:
                logging.info("[Training] END! @={}, best_epoch={}, best_loss_train={:.6f}" \
                            .format(time.time()-training_starttime, best_epoch, best_loss_train))
                break
        
        return {'enc_train_time': time.time()-training_starttime, \
            'enc_train_gpu': training_gpu_usage, \
            'enc_train_ram': training_ram_usage}
                
                
                
    def save_checkpoint(self):
        torch.save({'model_state_dict': self.model.state_dict(),},
                    self.checkpoint_file)
        return 
        






