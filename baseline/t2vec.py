import os
import time
import torch
import pandas as pd
import torch.nn as nn
from utils import tool_funcs
from datetime import datetime
from model_config import ModelConfig
from dataset_config import DatasetConfig
from torch.nn.utils import clip_grad_norm_
from utils.dataloader import read_traj_dataset
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

import logging
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().setLevel(logging.DEBUG)


class StackingGRUCell(nn.Module):
    """
    Multi-layer CRU Cell
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(StackingGRUCell, self).__init__()
        self.num_layers = num_layers
        self.grus = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        self.grus.append(nn.GRUCell(input_size, hidden_size))
        for i in range(1, num_layers):
            self.grus.append(nn.GRUCell(hidden_size, hidden_size))

    def forward(self, input, h0):
        """
        Input:
        input (batch, input_size): input tensor
        h0 (num_layers, batch, hidden_size): initial hidden state
        ---
        Output:
        output (batch, hidden_size): the final layer output tensor
        hn (num_layers, batch, hidden_size): the hidden state of each layer
        """
        hn = []
        output = input
        for i, gru in enumerate(self.grus):
            hn_i = gru(output, h0[i])
            hn.append(hn_i)
            if i != self.num_layers - 1:
                output = self.dropout(hn_i)
            else:
                output = hn_i
        hn = torch.stack(hn)
        return output, hn

class GlobalAttention(nn.Module):
    """
    $$a = \sigma((W_1 q)H)$$
    $$c = \tanh(W_2 [a H, q])$$
    """
    def __init__(self, hidden_size):
        super(GlobalAttention, self).__init__()
        self.L1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.L2 = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, q, H):
        """
        Input:
        q (batch, hidden_size): query
        H (batch, seq_len, hidden_size): context
        ---
        Output:
        c (batch, hidden_size)
        """
        # (batch, hidden_size) => (batch, hidden_size, 1)
        q1 = self.L1(q).unsqueeze(2)
        # (batch, seq_len)
        a = torch.bmm(H, q1).squeeze(2)
        a = self.softmax(a)
        # (batch, seq_len) => (batch, 1, seq_len)
        a = a.unsqueeze(1)
        # (batch, hidden_size)
        c = torch.bmm(a, H).squeeze(1)
        # (batch, hidden_size * 2)
        c = torch.cat([c, q], 1)
        return self.tanh(self.L2(c))

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout,
                       bidirectional, embedding):
        """
        embedding (vocab_size, input_size): pretrained embedding
        """
        super(Encoder, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        assert hidden_size % self.num_directions == 0
        self.hidden_size = hidden_size // self.num_directions
        self.num_layers = num_layers

        self.embedding = embedding
        self.rnn = nn.GRU(input_size, self.hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          dropout=dropout)

    def forward(self, input, lengths, h0=None):
        """
        Input:
        input (seq_len, batch): padded sequence tensor
        lengths (1, batch): sequence lengths
        h0 (num_layers*num_directions, batch, hidden_size): initial hidden state
        ---
        Output:
        hn (num_layers*num_directions, batch, hidden_size):
            the hidden state of each layer
        output (seq_len, batch, hidden_size*num_directions): output tensor
        """
        # (seq_len, batch) => (seq_len, batch, input_size)
        embed = self.embedding(input)
        lengths = lengths.data.view(-1).tolist()
        if lengths is not None:
            embed = pack_padded_sequence(embed, lengths)
        output, hn = self.rnn(embed, h0)
        if lengths is not None:
            output = pad_packed_sequence(output)[0]
        return hn, output

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, embedding):
        super(Decoder, self).__init__()
        self.embedding = embedding
        self.rnn = StackingGRUCell(input_size, hidden_size, num_layers,
                                   dropout)
        self.attention = GlobalAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, input, h, H, use_attention=True):
        """
        Input:
        input (seq_len, batch): padded sequence tensor
        h (num_layers, batch, hidden_size): input hidden state
        H (seq_len, batch, hidden_size): the context used in attention mechanism
            which is the output of encoder
        use_attention: If True then we use attention
        ---
        Output:
        output (seq_len, batch, hidden_size)
        h (num_layers, batch, hidden_size): output hidden state,
            h may serve as input hidden state for the next iteration,
            especially when we feed the word one by one (i.e., seq_len=1)
            such as in translation
        """
        assert input.dim() == 2, "The input should be of (seq_len, batch)"
        # (seq_len, batch) => (seq_len, batch, input_size)
        embed = self.embedding(input)
        output = []
        # split along the sequence length dimension
        for e in embed.split(1):
            e = e.squeeze(0) # (1, batch, input_size) => (batch, input_size)
            o, h = self.rnn(e, h)
            if use_attention:
                o = self.attention(o, H.transpose(0, 1))
            o = self.dropout(o)
            output.append(o)
        output = torch.stack(output)
        return output, h

class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size,
                       hidden_size, num_layers, dropout, bidirectional):
        super(EncoderDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        ## the embedding shared by encoder and decoder
        self.embedding = nn.Embedding(vocab_size, embedding_size,
                                      padding_idx=ModelConfig.t2vec.PAD)
        self.encoder = Encoder(embedding_size, hidden_size, num_layers,
                               dropout, bidirectional, self.embedding)
        self.decoder = Decoder(embedding_size, hidden_size, num_layers,
                               dropout, self.embedding)
        self.num_layers = num_layers

    def load_pretrained_embedding(self, path):
        if os.path.isfile(path):
            w = torch.load(path)
            self.embedding.weight.data.copy_(w)

    def encoder_hn2decoder_h0(self, h):
        """
        Input:
        h (num_layers * num_directions, batch, hidden_size): encoder output hn
        ---
        Output:
        h (num_layers, batch, hidden_size * num_directions): decoder input h0
        """
        if self.encoder.num_directions == 2:
            num_layers, batch, hidden_size = h.size(0)//2, h.size(1), h.size(2)
            return h.view(num_layers, 2, batch, hidden_size)\
                    .transpose(1, 2).contiguous()\
                    .view(num_layers, batch, hidden_size * 2)
        else:
            return h

    def forward(self, src, lengths, trg):
        """
        Input:
        src (src_seq_len, batch): source tensor
        lengths (1, batch): source sequence lengths
        trg (trg_seq_len, batch): target tensor, the `seq_len` in trg is not
            necessarily the same as that in src
        ---
        Output:
        output (trg_seq_len, batch, hidden_size)
        """
        encoder_hn, H = self.encoder(src, lengths)
        decoder_h0 = self.encoder_hn2decoder_h0(encoder_hn)
        ## for target we feed the range [BOS:EOS-1] into decoder
        output, decoder_hn = self.decoder(trg[:-1], decoder_h0, H)
        return output, decoder_h0
    
    
class t2vec_Trainer:
    def __init__(self):
        super(t2vec_Trainer, self).__init__()
        
        self.checkpoint_file_m0 = '{}/{}_m0_best.pt'.format(ModelConfig.t2vec.checkpoint_dir, DatasetConfig.dataset)
        self.checkpoint_file_m1 = '{}/{}_m1_best.pt'.format(ModelConfig.t2vec.checkpoint_dir, DatasetConfig.dataset)
        
        self.m0 = EncoderDecoder(ModelConfig.t2vec.vocab_size,
                            ModelConfig.t2vec.embedding_dim,
                            ModelConfig.t2vec.hidden_dim,
                            num_layers=ModelConfig.t2vec.num_layers,
                            dropout=ModelConfig.t2vec.dropout,
                            bidirectional=True).to(ModelConfig.device)
        self.m1 = nn.Sequential(nn.Linear(ModelConfig.t2vec.hidden_dim, ModelConfig.t2vec.vocab_size),
                        nn.LogSoftmax(dim=1)).to(ModelConfig.device)
        
        
    def train(self):
        training_starttime = time.time()
        train_dataset = read_traj_dataset(DatasetConfig.grid_total_file)
        train_dataloader = DataLoader(train_dataset, 
                                    batch_size=ModelConfig.t2vec.BATCH_SIZE, 
                                    shuffle=False, 
                                    num_workers=0, 
                                    drop_last=True)
        
        training_gpu_usage = training_ram_usage = 0.0

        criterion = self.NLLcriterion(ModelConfig.t2vec.vocab_size).to(ModelConfig.device)
        lossF = lambda o, t: criterion(o, t)
        
        m0_optimizer = torch.optim.Adam(self.m0.parameters(), ModelConfig.t2vec.learning_rate)
        m1_optimizer = torch.optim.Adam(self.m1.parameters(), ModelConfig.t2vec.learning_rate)
        
        logging.info("[Training] START! timestamp={}".format(datetime.fromtimestamp(training_starttime)))
        torch.autograd.set_detect_anomaly(True)
        
        best_loss_train = 100000
        best_epoch = 0
        bad_counter = 0
        bad_patience = ModelConfig.t2vec.training_bad_patience
        
        for i_ep in range(ModelConfig.t2vec.MAX_EPOCH):
            _time_ep = time.time()
            train_gpu = []
            train_ram = []
            
            self.m0.train()
            self.m1.train()
            
            loss = self.genLoss(self.m0, self.m1, train_dataloader, m0_optimizer, m1_optimizer, lossF)
            train_gpu.append(tool_funcs.GPUInfo.mem()[0])
            train_ram.append(tool_funcs.RAMInfo.mem())
            
            logging.info("[Training] ep={}: avg_loss={:.3f}, @={:.3f}/{:.3f}, gpu={}, ram={}" \
                    .format(i_ep, loss, time.time() - _time_ep, time.time() - training_starttime,
                    tool_funcs.GPUInfo.mem(), tool_funcs.RAMInfo.mem()))
            
            training_gpu_usage = tool_funcs.mean(train_gpu)
            training_ram_usage = tool_funcs.mean(train_ram)
            
            # early stopping
            if loss < best_loss_train:
                best_epoch = i_ep
                best_loss_train = loss
                bad_counter = 0
                self.save_checkpoint()
            else:
                bad_counter += 1
                
            if bad_counter == bad_patience or (i_ep + 1) == ModelConfig.t2vec.MAX_EPOCH:
                logging.info("[Training] END! @={}, best_epoch={}, best_loss_train={:.6f}" \
                            .format(time.time()-training_starttime, best_epoch, best_loss_train))
                break
        
        return {'enc_train_time': time.time()-training_starttime, \
            'enc_train_gpu': training_gpu_usage, \
            'enc_train_ram': training_ram_usage}
        
        

    def NLLcriterion(self, vocab_size):
        "construct NLL criterion"
        weight = torch.ones(vocab_size)
        weight[ModelConfig.t2vec.PAD] = 0
        ## The first dimension is not batch, thus we need
        ## to average over the batch manually
        #criterion = nn.NLLLoss(weight, size_average=False)
        criterion = nn.NLLLoss(weight, reduction='sum')
        return criterion

    def genLoss(self, m0, m1, train_loader, m0_optimizer, m1_optimizer, lossF):
        train_loss = 0
        for i_batch, batch in enumerate(train_loader):
            batch = batch.transpose(0,1).to(ModelConfig.device)
            eos = torch.full((1, batch.shape[1]), ModelConfig.t2vec.EOS).to(ModelConfig.device) # eos = 2500
            sos = torch.full((1, batch.shape[1]), ModelConfig.t2vec.BOS).to(ModelConfig.device) # sos = 2501
            src = torch.cat([batch, eos], dim=0)
            tgt = torch.cat([sos, batch], dim=0)
            #src (src_seq_len, batch): source tensor
            #lengths (1, batch): source sequence lengths
            src = src.long()
            lengths = torch.full((1, batch.shape[1]), batch.shape[0]).to(ModelConfig.device)
            tgt = tgt.long()
            output, _ = m0(src, lengths, tgt) # (seq_len, batch, hidden_size)
            ## we want to decode target in range [BOS+1:EOS]
            target = tgt[1:]
            ## (seq_len, generator_batch, hidden_size) =>
            ## (seq_len*generator_batch, hidden_size)
            o = output.view(-1, output.size(2)) # (seq_len*generator_batch, hidden_size)
            o = m1(o)
            ## (seq_len*generator_batch,)
            t = target.view(-1)
            m0_optimizer.zero_grad()
            m1_optimizer.zero_grad()
            loss = lossF(o, t)
            train_loss += loss
            loss.backward()
            clip_grad_norm_(m0.parameters(), ModelConfig.t2vec.max_grad_norm)
            clip_grad_norm_(m1.parameters(), ModelConfig.t2vec.max_grad_norm)
            m0_optimizer.step()
            m1_optimizer.step()
        return train_loss.div(len(train_loader.dataset))
    
    
    def encode(self):
        
        def encode_single(dataset_name, tp):
            logging.info('[{} {} Encode]start.'.format(dataset_name, tp))
            
            dataset = read_traj_dataset(DatasetConfig.dataset_folder+dataset_name
                                        +'/grid/{}_{}.pkl'.format(DatasetConfig.dataset_prefix, tp))
            dataloader = DataLoader(dataset=dataset,
                                    batch_size = ModelConfig.t2vec.BATCH_SIZE,
                                    shuffle = False,
                                    num_workers = 0,
                                    drop_last = True)
        
            index = {
                'prob': []
            }
        
            for _, batch in enumerate(dataloader):
                batch = batch.transpose(0,1).to(ModelConfig.device)
                eos = torch.full((1, batch.shape[1]), ModelConfig.t2vec.EOS).to(ModelConfig.device) # eos = 2500
                sos = torch.full((1, batch.shape[1]), ModelConfig.t2vec.BOS).to(ModelConfig.device) # sos = 2501
                src = torch.cat([batch, eos], dim=0)
                tgt = torch.cat([sos, batch], dim=0)
                src = src.long()
                lengths = torch.full((1, batch.shape[1]), batch.shape[0]).to(ModelConfig.device)
                tgt = tgt.long()
                _, decoder_h0 = self.m0(src, lengths, tgt)
                idx = decoder_h0.mean(dim=0)
                index['prob'].append(idx)
                
            index['prob'] = torch.cat(index['prob'], dim=0).view(-1, ModelConfig.t2vec.hidden_dim).cpu().detach().numpy()
            pd.DataFrame(index['prob']).to_csv(ModelConfig.t2vec.index_dir+'/{}/prob/{}_index.csv'.format(dataset_name, tp), header=None, index=None)
        
        
        db_size = [20,40,60,80,100] # dataset_size: 20K, 40K, 60K, 80K, 100K
        ds_rate = [0.3,0.4,0.5] # down-sampling rate: 
        dt_rate = [] # distort rate: 
        
        self.load_checkpoint()
        self.m0.eval()
        self.m1.eval()
        
        # original dataset
        dataset_name = 'train'
        self.make_indexfolder(dataset_name)
        encode_single(dataset_name, 'total')
        logging.info('[{} Encode]end.'.format(dataset_name))
        
        # for n_db in db_size:
        #     dataset_name = 'db_{}K'.format(n_db)
        #     self.make_indexfolder(dataset_name)
        #     encode_single(dataset_name, 'total')
        #     encode_single(dataset_name, 'ground')
        #     encode_single(dataset_name, 'test')
        #     logging.info('[{} Encode]end.'.format(dataset_name))
        
        # for v_ds in ds_rate:
        #     dataset_name = 'ds_{}'.format(v_ds)
        #     self.make_indexfolder(dataset_name)
        #     encode_single(dataset_name, 'total')
        #     encode_single(dataset_name, 'ground')
        #     encode_single(dataset_name, 'test')
        #     logging.info('[{} Encode]end.'.format(dataset_name))
            
    
    

    def save_checkpoint(self):
        torch.save({'m0_state_dict': self.m0.state_dict(),},
                    self.checkpoint_file_m0)
        torch.save({'m1_state_dict': self.m1.state_dict(),},
                    self.checkpoint_file_m1)
        return 
    
    def load_checkpoint(self):
        logging.info('[Load m0, m1...]')
        checkpoint_m0 = torch.load(self.checkpoint_file_m0)
        checkpoint_m1 = torch.load(self.checkpoint_file_m1)
        self.m0.load_state_dict(checkpoint_m0['m0_state_dict'])
        self.m1.load_state_dict(checkpoint_m1['m1_state_dict'])
        self.m0.to(ModelConfig.device)
        self.m1.to(ModelConfig.device)
        logging.info('[Load m0, m1] end.')
        return
    
    def make_indexfolder(self, dataset_name):
        folders = ['prob']
        base_dir = ModelConfig.t2vec.index_dir + '/{}/'.format(dataset_name)
        
        for folder in folders:
            os.makedirs(base_dir + folder, exist_ok=True)
        return