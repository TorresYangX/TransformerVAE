import os
import math
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


# Note:
# B: Batch size
# Ns: Source length
# Nt: Target length
# Nl: Latent length (typically = Ns)
# E: Embedding dimension
# H: Model dimension
# V: Vocab dimension


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
    

class Transformer(nn.Transformer):
    """
    A vanilla Transformer Encoder-Decoder in Pytorch

    Data format:
    SRC: ... [EOS]
    TGT: ... [EOS]
    Encoder_input(SRC): ... [EOS]
    Decoder_input(TGT): [SOS] ...

    For an autoencoder x -> x (SRC = TGT)
        The loss function requires SRC and logits.
    For different models x -> y (Eg: translation SRC != TGT)
        The loss function requires TGT and logits.

    If we keep this format the attention masks for padding are identical for autoencoder's encoder + decoder .
    """

    def __init__(self):
        super().__init__(
            d_model=ModelConfig.Transformer.embedding_dim,
            nhead=ModelConfig.Transformer.NUM_HEADS,
            num_encoder_layers=ModelConfig.Transformer.NUM_LAYERS,
            num_decoder_layers=ModelConfig.Transformer.NUM_LAYERS,
            dim_feedforward= ModelConfig.Transformer.DIM_FORWARD,
            dropout=ModelConfig.Transformer.dropout,
            batch_first=False,
            norm_first=False,
        )
        self.embedding = nn.Embedding(ModelConfig.Transformer.vocab_size, ModelConfig.Transformer.embedding_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(ModelConfig.Transformer.embedding_dim)
        self.output_proj = nn.Linear(ModelConfig.Transformer.embedding_dim, ModelConfig.Transformer.vocab_size)
        self.drop = nn.Dropout(ModelConfig.Transformer.dropout)

    def encode(self, src, src_key_padding_mask):
        """
        Encode the input ids to embeddings and then pass to the transformer encoder
        :param src: source token ids [Ns, B]
        :param src_key_padding_mask: Trues where to mask [B,Ns]
        :return: memory: [Ns,B,H]
        """
        # Add position encodings + Embeddings
        src = self.positional_encoding(self.drop(self.embedding(src.to(torch.int64)))).to(torch.float32)  # [Ns,B,H]

        # Transformer encoder
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)  # [Ns,B,H]
        return memory

    def latent_layer(self, encoder_output, src_key_padding_mask):
        """
        Latent layer for child classes like VAE

        :param encoder_output: encoder bov output [Ns,B,H]
        :param src_key_padding_mask: Trues where to mask [B,Nl] (typically encoder mask)
        :return: Z from the latent layer [Nl,B,H]
        """
        z = encoder_output  # [Ns,B,H]
        return {"z": z, "memory_key_padding_mask": src_key_padding_mask}  # [B,Nl]

    def decode(self, tgt, z, memory_key_padding_mask, tgt_key_padding_mask):
        """

        :param tgt: target token ids [Nt,B]
        :param z: output from the latent layer [Nl,B,H]
        :param memory_key_padding_mask: mask for latent layer [B, Nl] (typically Ns = Nl)
        :param tgt_key_padding_mask: target mask [B,Nt]
        :param args:
        :param kwargs:
        :return: logits over the vocabulary [Nt,B,V]
        """

        # Add position encodings + Embeddings
        tgt = self.positional_encoding(self.drop(self.embedding(tgt.to(torch.int64)))).to(torch.float32)  # [Nt,B,H]
        # Generate target teacher forcing mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(
            tgt.device
        )  # [Nt, Nt]
        output = self.decoder(
            tgt=tgt,  # [Nt,B,H]
            memory=z,  # [Nt,B,H]
            tgt_mask=tgt_mask,  # [Nt,Nt]
            tgt_key_padding_mask=tgt_key_padding_mask,  # [B,Nt]
            memory_key_padding_mask=memory_key_padding_mask,
        )  # [B,Nl]
        logits = self.output_proj(output)  # [Nt,B,V]
        return logits
    
    def loss_fn(self, logits, targets, z, memory_key_padding_mask):
        """
        Calculate the loss

        :param logits: output of the decoder [Nt,B,V]
        :param targets: target token ids [Nt, B]
        :return: Dictionary of scalar valued losses. With a value "Loss" to backprop averaged over batches.
        This is important as then the gradients are not dependent on B. However, want to log the loss over all data
        so we shouldn't average over batches as the average of averages is not the same thing when batches can be different sizes!
        https://lemire.me/blog/2005/10/28/average-of-averages-is-not-the-average/#:~:text=The%20average%20of%20averages%20is%20not%20the%20average,-A%20fact%20that&text=In%20fancy%20terms%2C%20the%20average,3

        """

        # Cross Entropy where pad = 0
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        # Transform targets
        targets = torch.flatten(targets)  # [Nt x B]
        # Transform vocabulary
        logits = torch.flatten(logits, start_dim=0, end_dim=1)  # [Nt x B, V]
        # Calculates loss and returns [Nt x B]
        cross_entropy_loss = criterion(logits.float(), targets.long())  # [Nt x B]
        # Average loss for backprop and sum loss for logging
        return {
            "Loss": torch.mean(cross_entropy_loss),
            "CrossEntropy": torch.sum(cross_entropy_loss),
        }

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        memory_mask=None,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Forward pass for all transformer models

        :param src: the sequence to the encoder (required). [Ns,B]
        :param tgt: the sequence  nce to the decoder (required). [Nt,B]
        :param src_mask: the additive mask for the src sequence (optional). [Ns, Ns]
        :param tgt_mask: the additive mask for the tgt sequence (optional). [Nt, Nt]
        :param memory_mask: the additive mask for the encoder output (optional). [Nt,Ns]
        :param src_key_padding_mask: the ByteTensor mask for src keys per batch (optional). [B,Ns]
        :param tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional). [B,Nt]
        :param memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).[B,Nl]
        :return: logits and latent dimension dictionary

        Check out here for more info masks on https://stackoverflow.com/questions/62170439/difference-between-src-mask-and-src-key-padding-mask
        The memory ones are interesting. I use memory_key_padding_mask to mask the tokens in the latent space.

        """

        # Encode
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)  # [Ns,B,H]
        # latent layer
        latent_output_dict = self.latent_layer(memory, src_key_padding_mask)
        # Decode
        output = self.decode(
            tgt=tgt,  # [Nt,B,H]
            z=latent_output_dict["z"],  # [Nl,B,H]
            tgt_key_padding_mask=tgt_key_padding_mask,  # [B,Nt]
            memory_key_padding_mask=latent_output_dict["memory_key_padding_mask"],
        )  # [B,Nl]

        return {
            "logits": output,  # [Nt, B, V]
            **latent_output_dict,
        }
    
    
class Transformer_Trainer:
    def __init__(self):
        super(Transformer_Trainer, self).__init__()
        
        self.model = Transformer().to(ModelConfig.device)
        
        self.eos_tensor = torch.full((ModelConfig.Transformer.Batch_size, 1), ModelConfig.Transformer.eos)
        self.sos_tensor = torch.full((ModelConfig.Transformer.Batch_size, 1), ModelConfig.Transformer.sos)
        
        self.src_key_padding_mask = torch.zeros((ModelConfig.Transformer.Batch_size, ModelConfig.Transformer.traj_len + 1), 
                                                dtype = torch.bool).to(ModelConfig.device)
        self.tgt_key_padding_mask = torch.zeros((ModelConfig.Transformer.Batch_size, ModelConfig.Transformer.traj_len + 1), 
                                                dtype = torch.bool).to(ModelConfig.device)
        
        self.checkpoint_file = '{}/{}_Trasformer_best.pt'.format(ModelConfig.Transformer.checkpoint_dir, DatasetConfig.dataset)
        
    def train(self):
        training_starttime = time.time()
        train_dataset = read_traj_dataset(DatasetConfig.grid_total_file)
        train_dataloader = DataLoader(train_dataset, 
                                        batch_size = ModelConfig.Transformer.Batch_size, 
                                        shuffle = False, 
                                        num_workers = 0, 
                                        drop_last = True)
        
        training_gpu_usage = training_ram_usage = 0.0
        logging.info("[Training] START! timestamp={}".format(datetime.fromtimestamp(training_starttime)))
        torch.autograd.set_detect_anomaly(True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr = ModelConfig.Transformer.learning_rate, weight_decay = 0.0001)
        
        best_loss_train = 100000
        best_epoch = 0
        bad_counter = 0
        bad_patience = ModelConfig.Transformer.training_bad_patience
        
        for i_ep in range(ModelConfig.Transformer.MAX_EPOCH):
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
                train_loss = self.model.loss_fn(**train_dict, targets = batch_tgt)
                
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
            
            if bad_counter == bad_patience or (i_ep + 1) == ModelConfig.Transformer.MAX_EPOCH:
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
                                    batch_size = ModelConfig.Transformer.Batch_size,
                                    shuffle = False,
                                    num_workers = 0,
                                    drop_last = True)
        elif tp == 'ground':
            ground_dataset = read_traj_dataset(DatasetConfig.grid_ground_file)
            dataloader = DataLoader(ground_dataset,
                                    batch_size = ModelConfig.Transformer.Batch_size,
                                    shuffle = False,
                                    num_workers = 0,
                                    drop_last = True)
        elif tp == 'test':
            test_dataset = read_traj_dataset(DatasetConfig.grid_test_file)
            dataloader = DataLoader(test_dataset,
                                    batch_size = ModelConfig.Transformer.Batch_size,
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
            batch_src = torch.cat([batch, self.eos_tensor], dim = 1).transpose(0,1).to(ModelConfig.device)
            batch_tgt = torch.cat([self.sos_tensor, batch], dim = 1).transpose(0,1).to(ModelConfig.device)
            
            enc_dict = self.model(batch_src, batch_tgt, self.src_key_padding_mask, self.tgt_key_padding_mask)
           
            encoder_ouput = enc_dict['z']
            prob = encoder_ouput.mean(dim=0, keepdim=True)
            index['prob'].append(prob)
            
        index['prob'] = torch.cat(index['prob'], dim = 0).view(-1, ModelConfig.Transformer.embedding_dim)
        
        np.savetxt(ModelConfig.Transformer.index_dir + '/prob/{}_prob.csv'.format(tp), index['prob'].cpu().detach().numpy())
        
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
        if not os.path.exists(ModelConfig.Transformer.index_dir+'/prob'):
            os.mkdir(ModelConfig.Transformer.index_dir+'/prob')
        return