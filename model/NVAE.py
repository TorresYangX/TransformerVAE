import os
import math
import torch
import numpy as np
import torch.nn as nn
from model_config import ModelConfig
from nvib.nvib_layer import Nvib
from nvib.kl import kl_dirichlet, kl_gaussian
from nvib.denoising_attention import DenoisingMultiheadAttention

device = ModelConfig.device
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


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
    n_epoch=ModelConfig.NVAE.MAX_EPOCH, type=ModelConfig.NVAE.KL_ANNEALING_GAUSSIAN
)
KL_ANNEALING_FACTOR_DIRICHLET_LIST = kl_annealing(
    n_epoch=ModelConfig.NVAE.MAX_EPOCH, type=ModelConfig.NVAE.KL_ANNEALING_DIRICHLET
)


class TokenEmbedding(nn.Module):
    def __init__(self):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=ModelConfig.NVAE.vocab_size, embedding_dim=ModelConfig.NVAE.embedding_dim)

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
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=ModelConfig.NVAE.embedding_dim, 
                                                                  nhead=ModelConfig.NVAE.nhead, 
                                                                  dim_feedforward=ModelConfig.NVAE.dim_forward, 
                                                                  dropout=ModelConfig.NVAE.dropout, 
                                                                  activation=ModelConfig.NVAE.activation, 
                                                                  batch_first=ModelConfig.NVAE.batch_first, 
                                                                  device=device) 
        self.TransformerEncoder = nn.TransformerEncoder(self.TransformerEncoderLayer, num_layers=ModelConfig.NVAE.num_layers)
        
    def forward(self, src, src_key_padding_mask):
        return self.TransformerEncoder(src, src_key_padding_mask = src_key_padding_mask)
    
class TransformerDecoder(nn.Module):
    def __init__(self):
        super(TransformerDecoder, self).__init__()
        self.TransformerDecoderLayer = nn.TransformerDecoderLayer(d_model=ModelConfig.NVAE.embedding_dim, 
                                                                  nhead=ModelConfig.NVAE.nhead, 
                                                                  dim_feedforward=ModelConfig.NVAE.dim_forward, 
                                                                  dropout=ModelConfig.NVAE.dropout, 
                                                                  activation=ModelConfig.NVAE.activation, 
                                                                  batch_first=ModelConfig.NVAE.batch_first, 
                                                                  device=device) 
        self.TransformerDecoder = nn.TransformerDecoder(self.TransformerDecoderLayer, num_layers=ModelConfig.NVAE.num_layers)
        for _, layer in enumerate(self.TransformerDecoder.layers):
            layer.multihead_attn = DenoisingMultiheadAttention(embed_dim=ModelConfig.NVAE.embedding_dim,
                                                               num_heads=ModelConfig.NVAE.nhead,
                                                               dropout=ModelConfig.NVAE.dropout,
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
        self.position_embedding = PositionalEncoding(ModelConfig.NVAE.embedding_dim)
        self.transformer_encoder = TransformerEncoder()
        self.nvib = Nvib(
            size_in = ModelConfig.NVAE.embedding_dim,
            size_out = ModelConfig.NVAE.embedding_dim,
            prior_mu = ModelConfig.NVAE.PRIOR_MU,
            prior_var = ModelConfig.NVAE.PRIOR_VAR,
            prior_alpha = ModelConfig.NVAE.PRIOR_ALPHA,
            kappa = ModelConfig.NVAE.KAPPA,
            delta = ModelConfig.NVAE.DELTA,
        )
        self.transformer_decoder = TransformerDecoder()
        self.output_proj = nn.Linear(ModelConfig.NVAE.embedding_dim, ModelConfig.NVAE.vocab_size)
        self.drop = nn.Dropout(ModelConfig.NVAE.dropout)

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
                prior_mu=ModelConfig.NVAE.PRIOR_MU,
                prior_var=ModelConfig.NVAE.PRIOR_VAR,
                kappa=ModelConfig.NVAE.KAPPA,
                **kwargs
            )
        )
        kl_loss_d = torch.mean(
            kl_dirichlet(
                prior_alpha=ModelConfig.NVAE.PRIOR_ALPHA,
                delta=ModelConfig.NVAE.DELTA,
                kappa=ModelConfig.NVAE.KAPPA,
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
            + ModelConfig.NVAE.KL_GAUSSIAN_LAMBDA * KL_ANNEALING_FACTOR_GAUSSIAN * kl_loss_g
            + ModelConfig.NVAE.KL_DIRICHLET_LAMBDA * KL_ANNEALING_FACTOR_DIRICHLET * kl_loss_d,
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
