import os
import time
import torch
import numpy as np
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


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.embedding_dim = ModelConfig.AE.embedding_dim
        self.trajectory_length = ModelConfig.AE.traj_len
        self.hidden_dim = ModelConfig.AE.hidden_dim
        self.latent_dim = ModelConfig.AE.latent_dim
        self.vocab_size = ModelConfig.AE.vocab_size
        self.BATCH_SIZE = ModelConfig.AE.BATCH_SIZE

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        self.encoder_lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.encoder = nn.Linear(self.hidden_dim, self.latent_dim)

        self.decoder_lstm = nn.LSTM(input_size=self.latent_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.decoder_fc = nn.Linear(self.hidden_dim, self.embedding_dim)
        self.outfc = nn.Linear( self.embedding_dim,  self.vocab_size)
        self.softmax = nn.Softmax(dim=2)

        self.hidden_init = (torch.autograd.Variable(torch.zeros(1, self.BATCH_SIZE, self.hidden_dim).to(ModelConfig.device)),
                            torch.autograd.Variable(torch.zeros(1, self.BATCH_SIZE, self.hidden_dim).to(ModelConfig.device)))

        self.softsign = nn.Softsign()

    def encode(self, x):
        x = self.embedding(x.to(torch.int64))
        h, _ = self.encoder_lstm(x, self.hidden_init)
        h = h[:, -1:, :]
        h = self.softsign(self.encoder(h))
        return h

    def decode(self, h):
        h = h.repeat(1, self.trajectory_length, 1)
        h, _ = self.decoder_lstm(h, self.hidden_init)
        x_hat = self.softsign(self.decoder_fc(h))
        x_hat = self.outfc(x_hat)
        x_hat = self.softmax(x_hat)
        return x_hat
    
    def loss_fn(self, logits, targets, prob):
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        targets = torch.flatten(targets)
        logits = torch.flatten(logits, start_dim=0, end_dim=1)
        BCE = criterion(logits.float(), targets.long()).mean()
        return {
            "Loss": BCE,
            "CrossEntropy": BCE,
        }

    def forward(self, x):
        h = self.encode(x)
        x_hat = self.decode(h)
        return {
            'logits': x_hat,
            'prob': h
        }
        
        
class AE_Trainer:
    def __init__(self):
        super(AE_Trainer, self).__init__()
        
        self.model = AE().to(ModelConfig.device)
        self.checkpoint_file = '{}/{}_AE_best.pt'.format(ModelConfig.AE.checkpoint_dir, DatasetConfig.dataset)


    def train(self):
        training_starttime = time.time()
        train_dataset = read_traj_dataset(DatasetConfig.grid_total_file)
        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=ModelConfig.AE.BATCH_SIZE, 
                                      shuffle=False, 
                                      num_workers=0, 
                                      drop_last=True)
        
        training_gpu_usage = training_ram_usage = 0.0
        logging.info("[Training] START! timestamp={}".format(datetime.fromtimestamp(training_starttime)))
        torch.autograd.set_detect_anomaly(True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr = ModelConfig.AE.learning_rate)
        
        best_loss_train = 100000
        best_epoch = 0
        bad_counter = 0
        bad_patience = ModelConfig.AE.training_bad_patience
        
        for i_ep in range(ModelConfig.AE.MAX_EPOCH):
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
                
            if bad_counter == bad_patience or (i_ep + 1) == ModelConfig.AE.MAX_EPOCH:
                logging.info("[Training] END! @={}, best_epoch={}, best_loss_train={:.6f}" \
                            .format(time.time()-training_starttime, best_epoch, best_loss_train))
                break
        
        return {'enc_train_time': time.time()-training_starttime, \
            'enc_train_gpu': training_gpu_usage, \
            'enc_train_ram': training_ram_usage}
        
    
    def encode(self, tp):
        if tp == 'total':
            total_dataset = read_traj_dataset(DatasetConfig.grid_total_file)
            dataloader = DataLoader(total_dataset,
                                    batch_size = ModelConfig.AE.BATCH_SIZE,
                                    shuffle = False,
                                    num_workers = 0,
                                    drop_last = True)
        elif tp == 'ground':
            ground_dataset = read_traj_dataset(DatasetConfig.grid_ground_file)
            dataloader = DataLoader(ground_dataset,
                                    batch_size = ModelConfig.AE.BATCH_SIZE,
                                    shuffle = False,
                                    num_workers = 0,
                                    drop_last = True)
        elif tp == 'test':
            test_dataset = read_traj_dataset(DatasetConfig.grid_test_file)
            dataloader = DataLoader(test_dataset,
                                    batch_size = ModelConfig.AE.BATCH_SIZE,
                                    shuffle = False,
                                    num_workers = 0,
                                    drop_last = True)
        else:
            raise ValueError("Invalid type of dataset.")
        
        logging.info('[Encode]start.')
        self.make_indexfolder()
        self.load_checkpoint()
        self.model.eval()
        
        index = {
            'prob': [],
        }
        
        for i_batch, batch in enumerate(dataloader):
            dict = self.model(batch.to(ModelConfig.device))
            prob = dict['prob'][:, 0, :]
            index['prob'].append(prob)
            
        index['prob'] = torch.cat(index['prob'], dim = 0).view(-1, ModelConfig.AE.latent_dim)
        
        np.savetxt(ModelConfig.AE.index_dir + '/prob/{}_prob.csv'.format(tp), index['prob'].cpu().detach().numpy())
        
        return
            
                
                
                
    def save_checkpoint(self):
        torch.save({'model_state_dict': self.model.state_dict(),},
                    self.checkpoint_file)
        return  
    
    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return
    
    def make_indexfolder(self):
        if not os.path.exists(ModelConfig.AE.index_dir+'/prob'):
            os.mkdir(ModelConfig.AE.index_dir+'/prob')
        return