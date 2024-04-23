import os
import time
import torch
import pandas as pd
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
        
        self.eos_tensor = torch.full((ModelConfig.NVAE.BATCH_SIZE, 1), ModelConfig.NVAE.eos)
        self.sos_tensor = torch.full((ModelConfig.NVAE.BATCH_SIZE, 1), ModelConfig.NVAE.sos)
        
        self.src_key_padding_mask = torch.zeros((ModelConfig.NVAE.BATCH_SIZE, ModelConfig.NVAE.traj_len + 1), 
                                                dtype = torch.bool).to(ModelConfig.device)
        self.tgt_key_padding_mask = torch.zeros((ModelConfig.NVAE.BATCH_SIZE, ModelConfig.NVAE.traj_len + 1), 
                                                dtype = torch.bool).to(ModelConfig.device)

        self.checkpoint_file = '{}/{}_NVAE_best.pt'.format(ModelConfig.NVAE.checkpoint_dir, 
                                                           DatasetConfig.dataset)

        
    def train(self):
        training_starttime = time.time()
        train_dataset = read_traj_dataset(DatasetConfig.grid_total_file)
        train_dataloader = DataLoader(train_dataset, 
                                        batch_size = ModelConfig.NVAE.BATCH_SIZE, 
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
    def encode(self):
        """
        1. read best model
        2. read trajs from file, then -> embeddings
        n. varying db size, downsampling rates, and distort rates
        """
        
        def encode_single(dataset_name, tp):
            logging.info('[{} {} Encode]start.'.format(dataset_name, tp))
            
            dataset = read_traj_dataset(DatasetConfig.dataset_folder+dataset_name
                                        +'/grid/{}_{}.pkl'.format(DatasetConfig.dataset_prefix, tp))
            dataloader = DataLoader(dataset=dataset,
                                    batch_size = ModelConfig.NVAE.BATCH_SIZE,
                                    shuffle = False,
                                    num_workers = 0,
                                    drop_last = True)
            
            self.make_indexfolder(dataset_name)
            self.load_checkpoint()
            self.model.train()
            
            index = {
                'mu': [],
                'logvar': [],
                'pi': [],
                'alpha': []
            }

            for _, batch in enumerate(dataloader):
                batch_src = torch.cat([batch, self.eos_tensor], dim = 1).transpose(0,1).to(ModelConfig.device)
                batch_tgt = torch.cat([self.sos_tensor, batch], dim = 1).transpose(0,1).to(ModelConfig.device)
                
                enc_dict = self.model(batch_src, batch_tgt, self.src_key_padding_mask, self.tgt_key_padding_mask)
                mu = enc_dict['mu'].mean(dim = 0, keepdim = True).cpu().detach()
                logvar = enc_dict['logvar'].mean(dim = 0, keepdim = True).cpu().detach()
                pi = enc_dict['pi'].repeat(1,1,ModelConfig.NVAE.embedding_dim).mean(dim = 0, keepdim = True).cpu().detach()
                alpha = enc_dict['alpha'].repeat(1,1,ModelConfig.NVAE.embedding_dim).mean(dim = 0, keepdim = True).cpu().detach()
                
                index['mu'].append(mu)
                index['logvar'].append(logvar)
                index['pi'].append(pi)
                index['alpha'].append(alpha)
                
            index['mu'] = torch.cat(index['mu'], dim = 0).view(-1, ModelConfig.NVAE.embedding_dim)
            index['logvar'] = torch.cat(index['logvar'], dim = 0).view(-1, ModelConfig.NVAE.embedding_dim)
            index['pi'] = torch.cat(index['pi'], dim = 0).view(-1, ModelConfig.NVAE.embedding_dim)
            index['alpha'] = torch.cat(index['alpha'], dim = 0).view(-1, ModelConfig.NVAE.embedding_dim)
            
            
            
            pd.DataFrame(index['mu']).to_csv(ModelConfig.NVAE.index_dir+'/{}/mu/{}_index.csv'.format(dataset_name, tp), header=None, index=None)
            pd.DataFrame(index['logvar']).to_csv(ModelConfig.NVAE.index_dir+'/{}/logvar/{}_index.csv'.format(dataset_name, tp), header=None, index=None)
            pd.DataFrame(index['pi']).to_csv(ModelConfig.NVAE.index_dir+'/{}/pi/{}_index.csv'.format(dataset_name, tp), header=None, index=None)
            pd.DataFrame(index['alpha']).to_csv(ModelConfig.NVAE.index_dir+'/{}/alpha/{}_index.csv'.format(dataset_name, tp), header=None, index=None)
        
            return
        
        db_size = [20,40,60,80,100] # dataset_size: 20K, 40K, 60K, 80K, 100K
        ds_rate = [] # down-sampling rate: 
        dt_rate = [] # distort rate: 
        for n_db in db_size:
            dataset_name = 'db_{}K'.format(n_db)
            encode_single(dataset_name, 'total')
            encode_single(dataset_name, 'ground')
            encode_single(dataset_name, 'test')
            logging.info('[{} Encode]end.'.format(dataset_name))
            
        

                      
              
    def save_checkpoint(self):
        torch.save({'model_state_dict': self.model.state_dict(),},
                    self.checkpoint_file)
        return  
    
    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(ModelConfig.device)
        return
    
    def make_indexfolder(self, dataset_name):
        folders = ['mu', 'logvar', 'pi', 'alpha']
        base_dir = ModelConfig.NVAE.index_dir + '/{}/'.format(dataset_name)
        
        for folder in folders:
            os.makedirs(base_dir + folder, exist_ok=True)
        return