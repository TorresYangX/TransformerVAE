import os
import time
import torch
import numpy as np
from utils import tool_funcs
from datetime import datetime
from model_config import ModelConfig
from model.NVAE import TransformerNvib
from dataset_config import DatasetConfig
from utils.dataloader import read_traj_dataset
from torch.utils.data.dataloader import DataLoader

import logging
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().setLevel(logging.DEBUG)


class Trainer:
    def __init__(self):
        super(Trainer, self).__init__()
        
        self.model = TransformerNvib().to(ModelConfig.device)
        
        self.eos_tensor = torch.full((ModelConfig.NVAE.Batch_size, 1), ModelConfig.NVAE.eos)
        self.sos_tensor = torch.full((ModelConfig.NVAE.Batch_size, 1), ModelConfig.NVAE.sos)
        
        self.src_key_padding_mask = torch.zeros((ModelConfig.NVAE.Batch_size, ModelConfig.NVAE.traj_len + 1), 
                                                dtype = torch.bool).to(ModelConfig.device)
        self.tgt_key_padding_mask = torch.zeros((ModelConfig.NVAE.Batch_size, ModelConfig.NVAE.traj_len + 1), 
                                                dtype = torch.bool).to(ModelConfig.device)

        self.checkpoint_file = '{}/{}_NVAE_best.pt'.format(ModelConfig.NVAE.checkpoint_dir, 
                                                           DatasetConfig.dataset)

        
    def train(self):
        training_starttime = time.time()
        train_dataset = read_traj_dataset(DatasetConfig.grid_total_file)
        train_dataloader = DataLoader(train_dataset, 
                                            batch_size = ModelConfig.NVAE.Batch_size, 
                                            shuffle = False, 
                                            num_workers = 0, 
                                            drop_last = True)
        
        training_gpu_usage = training_ram_usage = 0.0
        logging.info("[Training] START! timestamp={}".format(datetime.fromtimestamp(training_starttime)))
        torch.autograd.set_detect_anomaly(True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr = ModelConfig.NVAE.learning_rate, weight_decay = 0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = ModelConfig.NVAE.training_lr_degrade_step, 
                                                    gamma = ModelConfig.NVAE.training_lr_degrade_gamma)

        best_loss_train = 100000
        best_epoch = 0
        bad_counter = 0
        bad_patience = ModelConfig.NVAE.training_bad_patience
        
        for i_ep in range(ModelConfig.NVAE.MAX_EPOCH):
            _time_ep = time.time()
            loss_ep = []
            train_gpu = []
            train_ram = []
            
            self.model.train()
            
            _time_batch_start = time.time()
            for i_batch, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                batch_src = torch.cat([batch, self.eos_tensor], dim = 1).transpose(0,1).to(ModelConfig.device)
                batch_tgt = torch.cat([self.sos_tensor, batch], dim = 1).transpose(0,1).to(ModelConfig.device)
                
                train_dict = self.model(batch_src, batch_tgt, self.src_key_padding_mask, self.tgt_key_padding_mask)
                train_loss = self.model.loss(**train_dict, targets = batch_tgt, epoch = i_ep)
                
                (train_loss['Loss'] / ModelConfig.NVAE.ACCUMULATION_STEPS).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                
                if ((i_batch + 1) % ModelConfig.NVAE.ACCUMULATION_STEPS == 0) or (i_batch + 1 == len(train_dataloader)):
                    optimizer.step()
                    optimizer.zero_grad()
                loss_ep.append(train_loss['Loss'].item())
                train_gpu.append(tool_funcs.GPUInfo.mem()[0])
                train_ram.append(tool_funcs.RAMInfo.mem())
                
                if i_batch % 100 == 0 and i_batch:
                    logging.debug("[Training] ep-batch={}-{}, loss={:.3f}, @={:.3f}, gpu={}, ram={}" \
                            .format(i_ep, i_batch, train_loss['Loss'].item(), time.time() - _time_batch_start,
                                    tool_funcs.GPUInfo.mem(), tool_funcs.RAMInfo.mem()))
                
            scheduler.step() # decay before optimizer when pytorch < 1.1
                
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
            
            if bad_counter == bad_patience or (i_ep + 1) == ModelConfig.NVAE.MAX_EPOCH:
                logging.info("[Training] END! @={}, best_epoch={}, best_loss_train={:.6f}" \
                            .format(time.time()-training_starttime, best_epoch, best_loss_train))
                break
            
        return {'enc_train_time': time.time()-training_starttime, \
            'enc_train_gpu': training_gpu_usage, \
            'enc_train_ram': training_ram_usage}
        
    @torch.no_grad()
    def encode(self, tp):
        """
        1. read best model
        2. read trajs from file, then -> embeddings
        n. varying db size, downsampling rates, and distort rates
        """
        if tp == 'total':
            total_dataset = read_traj_dataset(DatasetConfig.grid_total_file)
            dataloader = DataLoader(total_dataset,
                                    batch_size = ModelConfig.NVAE.Batch_size,
                                    shuffle = False,
                                    num_workers = 0,
                                    drop_last = True)
        elif tp == 'ground':
            ground_dataset = read_traj_dataset(DatasetConfig.grid_ground_file)
            dataloader = DataLoader(ground_dataset,
                                    batch_size = ModelConfig.NVAE.Batch_size,
                                    shuffle = False,
                                    num_workers = 0,
                                    drop_last = True)
        elif tp == 'test':
            test_dataset = read_traj_dataset(DatasetConfig.grid_test_file)
            dataloader = DataLoader(test_dataset,
                                    batch_size = ModelConfig.NVAE.Batch_size,
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
            'mu': [],
            'logvar': [],
            'pi': [],
            'alpha': []
        }

        for i_batch, batch in enumerate(dataloader):
            batch_src = torch.cat([batch, self.eos_tensor], dim = 1).transpose(0,1).to(ModelConfig.device)
            batch_tgt = torch.cat([self.sos_tensor, batch], dim = 1).transpose(0,1).to(ModelConfig.device)
            
            enc_dict = self.model(batch_src, batch_tgt, self.src_key_padding_mask, self.tgt_key_padding_mask)
            mu = enc_dict['mu'].mean(dim = 0, keepdim = True)
            logvar = enc_dict['logvar'].mean(dim = 0, keepdim = True)
            pi = enc_dict['pi'].repeat(1,1,ModelConfig.NVAE.embedding_dim).mean(dim = 0, keepdim = True)
            alpha = enc_dict['alpha'].repeat(1,1,ModelConfig.NVAE.embedding_dim).mean(dim = 0, keepdim = True)
            
            index['mu'].append(mu)
            index['logvar'].append(logvar)
            index['pi'].append(pi)
            index['alpha'].append(alpha)
            
        index['mu'] = torch.cat(index['mu'], dim = 0).view(-1, ModelConfig.NVAE.embedding_dim)
        index['logvar'] = torch.cat(index['logvar'], dim = 0).view(-1, ModelConfig.NVAE.embedding_dim)
        index['pi'] = torch.cat(index['pi'], dim = 0).view(-1, ModelConfig.NVAE.embedding_dim)
        index['alpha'] = torch.cat(index['alpha'], dim = 0).view(-1, ModelConfig.NVAE.embedding_dim)
        
        np.savetxt(ModelConfig.NVAE.index_dir + '/mu/{}_mu.csv'.format(tp), index['mu'].cpu().detach().numpy())
        np.savetxt(ModelConfig.NVAE.index_dir + '/logvar/{}_sigma.csv'.format(tp), index['logvar'].cpu().detach().numpy())
        np.savetxt(ModelConfig.NVAE.index_dir + '/pi/{}_pi.csv'.format(tp), index['pi'].cpu().detach().numpy())
        np.savetxt(ModelConfig.NVAE.index_dir + '/alpha/{}_alpha.csv'.format(tp), index['alpha'].cpu().detach().numpy())
    
        return

                      
              
    def save_checkpoint(self):
        torch.save({'model_state_dict': self.model.state_dict(),},
                    self.checkpoint_file)
        return  
    
    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(ModelConfig.device)
        return
    
    def make_indexfolder(self):
        if not os.path.exists(ModelConfig.NVAE.index_dir+'/mu'):
            os.mkdir(ModelConfig.NVAE.index_dir+'/mu')
            os.mkdir(ModelConfig.NVAE.index_dir+'/logvar')
            os.mkdir(ModelConfig.NVAE.index_dir+'/pi')
            os.mkdir(ModelConfig.NVAE.index_dir+'/alpha')
        return