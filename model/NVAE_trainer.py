import os
import time
import torch
import numpy as np
from config import Config
from utils import tool_funcs
from datetime import datetime
from model.NVAE import TransformerNvib
from utils.dataloader import read_traj_dataset
from torch.utils.data.dataloader import DataLoader

import logging
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().setLevel(logging.DEBUG)


class Trainer:
    def __init__(self):
        super(Trainer, self).__init__()
        
        self.model = TransformerNvib().to(Config.device)
        
        self.eos_tensor = torch.full((Config.Batch_size, 1), Config.eos)
        self.sos_tensor = torch.full((Config.Batch_size, 1), Config.sos)
        
        self.src_key_padding_mask = torch.zeros((Config.Batch_size, Config.traj_len + 1), dtype = torch.bool).to(Config.device)
        self.tgt_key_padding_mask = torch.zeros((Config.Batch_size, Config.traj_len + 1), dtype = torch.bool).to(Config.device)

        self.checkpoint_file = '{}/{}_NVAE_best.pt'.format(Config.checkpoint_dir, Config.dataset)

        
    def train(self):
        training_starttime = time.time()
        train_dataset = read_traj_dataset(Config.grid_total_file)
        train_dataloader = DataLoader(train_dataset, 
                                            batch_size = Config.Batch_size, 
                                            shuffle = False, 
                                            num_workers = 0, 
                                            drop_last = True)
        
        training_gpu_usage = training_ram_usage = 0.0
        logging.info("[Training] START! timestamp={}".format(datetime.fromtimestamp(training_starttime)))
        torch.autograd.set_detect_anomaly(True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr = Config.learning_rate, weight_decay = 0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = Config.training_lr_degrade_step, gamma = Config.training_lr_degrade_gamma)

        best_loss_train = 100000
        best_epoch = 0
        bad_counter = 0
        bad_patience = Config.training_bad_patience
        
        for i_ep in range(Config.MAX_EPOCH):
            _time_ep = time.time()
            loss_ep = []
            train_gpu = []
            train_ram = []
            
            self.model.train()
            
            _time_batch_start = time.time()
            for i_batch, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                batch_src = torch.cat([batch, self.eos_tensor], dim = 1).transpose(0,1).to(Config.device)
                batch_tgt = torch.cat([self.sos_tensor, batch], dim = 1).transpose(0,1).to(Config.device)
                
                train_dict = self.model(batch_src, batch_tgt, self.src_key_padding_mask, self.tgt_key_padding_mask)
                train_loss = self.model.loss(**train_dict, targets = batch_tgt, epoch = i_ep)
                
                (train_loss['Loss'] / Config.ACCUMULATION_STEPS).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                
                if ((i_batch + 1) % Config.ACCUMULATION_STEPS == 0) or (i_batch + 1 == len(train_dataloader)):
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
            
            if bad_counter == bad_patience or (i_ep + 1) == Config.MAX_EPOCH:
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
            total_dataset = read_traj_dataset(Config.grid_total_file)
            dataloader = DataLoader(total_dataset,
                                    batch_size = Config.Batch_size,
                                    shuffle = False,
                                    num_workers = 0,
                                    drop_last = True)
        elif tp == 'ground':
            ground_dataset = read_traj_dataset(Config.grid_ground_file)
            dataloader = DataLoader(ground_dataset,
                                    batch_size = Config.Batch_size,
                                    shuffle = False,
                                    num_workers = 0,
                                    drop_last = True)
        elif tp == 'test':
            test_dataset = read_traj_dataset(Config.grid_test_file)
            dataloader = DataLoader(test_dataset,
                                    batch_size = Config.Batch_size,
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
            batch_src = torch.cat([batch, self.eos_tensor], dim = 1).transpose(0,1).to(Config.device)
            batch_tgt = torch.cat([self.sos_tensor, batch], dim = 1).transpose(0,1).to(Config.device)
            
            enc_dict = self.model(batch_src, batch_tgt, self.src_key_padding_mask, self.tgt_key_padding_mask)
            mu = enc_dict['mu'].mean(dim = 0, keepdim = True)
            logvar = enc_dict['logvar'].mean(dim = 0, keepdim = True)
            pi = enc_dict['pi'].repeat(1,1,Config.embedding_dim).mean(dim = 0, keepdim = True)
            alpha = enc_dict['alpha'].repeat(1,1,Config.embedding_dim).mean(dim = 0, keepdim = True)
            
            index['mu'].append(mu)
            index['logvar'].append(logvar)
            index['pi'].append(pi)
            index['alpha'].append(alpha)
            
        index['mu'] = torch.cat(index['mu'], dim = 0).view(-1, Config.embedding_dim)
        index['logvar'] = torch.cat(index['logvar'], dim = 0).view(-1, Config.embedding_dim)
        index['pi'] = torch.cat(index['pi'], dim = 0).view(-1, Config.embedding_dim)
        index['alpha'] = torch.cat(index['alpha'], dim = 0).view(-1, Config.embedding_dim)
        
        np.savetxt(Config.index_dir + '/mu/{}_mu.csv'.format(tp), index['mu'].cpu().detach().numpy())
        np.savetxt(Config.index_dir + '/logvar/{}_sigma.csv'.format(tp), index['logvar'].cpu().detach().numpy())
        np.savetxt(Config.index_dir + '/pi/{}_pi.csv'.format(tp), index['pi'].cpu().detach().numpy())
        np.savetxt(Config.index_dir + '/alpha/{}_alpha.csv'.format(tp), index['alpha'].cpu().detach().numpy())
    
        return

                      
              
    def save_checkpoint(self):
        torch.save({'model_state_dict': self.model.state_dict(),},
                    self.checkpoint_file)
        return  
    
    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(Config.device)
        return
    
    def make_indexfolder(self):
        if not os.path.exists(Config.index_dir+'/mu'):
            os.mkdir(Config.index_dir+'/mu')
            os.mkdir(Config.index_dir+'/logvar')
            os.mkdir(Config.index_dir+'/pi')
            os.mkdir(Config.index_dir+'/alpha')
        return
    
    # def __init__(self, model, optimizer, train_loader, test_loader, trajectory_length, grid_num, epoch, ACCUMULATION_STEPS):
    #     self.model = model
    #     self.optimizer = optimizer
    #     self.train_loader = train_loader
    #     self.test_loader = test_loader
    #     self.trajectory_length = trajectory_length
    #     self.grid_num = grid_num
    #     self.epoch = epoch
    #     self.ACCUMULATION_STEPS = ACCUMULATION_STEPS
    
    # def training(self):
    #     train_losses_value = 0
    #     for idx, x in enumerate(self.train_loader):
    #         x = x[0].transpose(0,1).to(device)
    #         eos = torch.full((1, x.shape[1]), self.grid_num*self.grid_num).to(device)
    #         sos = torch.full((1, x.shape[1]), self.grid_num*self.grid_num+1).to(device)
    #         src = torch.cat([x, eos], dim=0)
    #         tgt = torch.cat([sos, x], dim=0)

    #         src_key_padding_mask = torch.zeros((x.shape[1], self.trajectory_length + 1), dtype=torch.bool).to(device)
    #         tgt_key_padding_mask = torch.zeros((x.shape[1], self.trajectory_length + 1), dtype=torch.bool).to(device)

    #         train_outputs_dict = self.model(
    #             src,
    #             tgt,
    #             src_key_padding_mask=src_key_padding_mask,
    #             tgt_key_padding_mask=tgt_key_padding_mask,
    #         )
    #         train_losses = self.model.loss(**train_outputs_dict, targets=tgt, epoch=self.epoch)
    #         (train_losses["Loss"] / self.ACCUMULATION_STEPS).backward()
    #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
    #         if ((idx + 1) % self.ACCUMULATION_STEPS == 0) or (idx + 1 == len(self.train_loader)):
    #             self.optimizer.step()
    #             self.optimizer.zero_grad()
    #         train_losses_value += train_losses["Loss"].item()
    #     return train_losses_value / len(self.train_loader.dataset)

    # def evaluation(self):
    #     test_losses_value = 0
    #     for _, x in enumerate(self.test_loader):
    #         with torch.no_grad():
    #             x = x[0].transpose(0,1).to(device)
    #             eos = torch.full((1, x.shape[1]), self.grid_num*self.grid_num).to(device)
    #             sos = torch.full((1, x.shape[1]), self.grid_num*self.grid_num+1).to(device)
    #             src = torch.cat([x, eos], dim=0)
    #             tgt = torch.cat([sos, x], dim=0)

    #             src_key_padding_mask = torch.zeros((x.shape[1], self.trajectory_length + 1), dtype=torch.bool).to(device)
    #             tgt_key_padding_mask = torch.zeros((x.shape[1], self.trajectory_length + 1), dtype=torch.bool).to(device)

    #             test_outputs_dict = self.model(
    #                 src,
    #                 tgt,
    #                 src_key_padding_mask=src_key_padding_mask,
    #                 tgt_key_padding_mask=tgt_key_padding_mask,
    #             )
    #             test_losses = self.model.loss(**test_outputs_dict, targets=tgt, epoch=self.epoch)
    #             test_losses_value += test_losses["Loss"].item()
    #     return test_losses_value / len(self.test_loader.dataset)