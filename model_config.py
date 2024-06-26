import os
import torch
from dataset_config import DatasetConfig


root_dir = os.path.abspath(__file__)[:-15]

class ModelConfig:
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    class NVAE:
        model = 'NVAE'
        
        traj_len = 60
        grid_num = 50
        checkpoint_dir = root_dir + '/exp/{}/NVAE'.format(DatasetConfig.dataset)
        index_dir = root_dir + '/exp/{}/NVAE/index'.format(DatasetConfig.dataset)
        
        vocab_size = grid_num * grid_num + 2 #V
        sos = grid_num * grid_num
        eos = grid_num * grid_num + 1
        BATCH_SIZE = 16 #B
        d_model = 64
        embedding_dim = 16 # H
        dim_forward = 256
        nhead = 1
        dropout = 0.1
        activation = 'relu'
        batch_first = False
        num_layers = 6
        PRIOR_MU = 0
        PRIOR_VAR = 1
        PRIOR_ALPHA = 1
        KAPPA = 1
        DELTA = 1
        KL_GAUSSIAN_LAMBDA = 0.001
        KL_DIRICHLET_LAMBDA = 1
        KL_ANNEALING_GAUSSIAN = "constant"
        KL_ANNEALING_DIRICHLET = "constant"
        
        learning_rate = 1e-4
        training_lr_degrade_step = 5
        training_lr_degrade_gamma = 0.5
        training_bad_patience = 5
        MAX_EPOCH = 500
        ACCUMULATION_STEPS = 1
        
        @classmethod
        def to_str(cls): # __str__, self
            dic = cls.__dict__.copy()
            lst = list(filter( \
                            lambda p: (not p[0].startswith('__')) and type(p[1]) != classmethod, \
                            dic.items() \
                            ))
            return '\n'.join([str(k) + ' = ' + str(v) for k, v in lst])
        
    class AE:
        model = 'AE'
        
        embedding_dim = 64
        hidden_dim = 32
        latent_dim = 16
        traj_len = 60
        grid_num = 50
        vocab_size = grid_num * grid_num + 2
        dropout = 0.1
        BATCH_SIZE = 16
        
        learning_rate = 1e-5
        training_bad_patience = 10
        MAX_EPOCH = 500
        checkpoint_dir = 'exp/{}/AE'.format(DatasetConfig.dataset)
        index_dir = root_dir + '/exp/{}/AE/index'.format(DatasetConfig.dataset)
        
        @classmethod
        def to_str(cls): # __str__, self
            dic = cls.__dict__.copy()
            lst = list(filter( \
                            lambda p: (not p[0].startswith('__')) and type(p[1]) != classmethod, \
                            dic.items() \
                            ))
            return '\n'.join([str(k) + ' = ' + str(v) for k, v in lst])
        
        
    class VAE:
        model = 'VAE'
        
        embedding_dim = 64
        hidden_dim = 32
        latent_dim = 16
        traj_len = 60
        grid_num = 50
        vocab_size = grid_num * grid_num + 2
        BATCH_SIZE = 16
        KL_WEIGHT = 0.01
        
        learning_rate = 1e-5
        training_bad_patience = 10
        MAX_EPOCH = 500
        checkpoint_dir = 'exp/{}/VAE'.format(DatasetConfig.dataset)
        index_dir = root_dir + '/exp/{}/VAE/index'.format(DatasetConfig.dataset)
        
        @classmethod
        def to_str(cls): # __str__, self
            dic = cls.__dict__.copy()
            lst = list(filter( \
                            lambda p: (not p[0].startswith('__')) and type(p[1]) != classmethod, \
                            dic.items() \
                            ))
            return '\n'.join([str(k) + ' = ' + str(v) for k, v in lst])
        
        
    class Transformer:
        model = 'Transformer'
        
        embedding_dim = 16
        NUM_HEADS = 8
        NUM_LAYERS = 6
        DIM_FORWARD = 512
        dropout = 0.1
        traj_len = 60
        grid_num = 50
        vocab_size = grid_num * grid_num + 2
        sos = grid_num * grid_num
        eos = grid_num * grid_num + 1
        BATCH_SIZE = 16
        
        learning_rate = 1e-3
        training_bad_patience = 10
        MAX_EPOCH = 500
        checkpoint_dir = 'exp/{}/Transformer'.format(DatasetConfig.dataset)
        index_dir = root_dir + '/exp/{}/Transformer/index'.format(DatasetConfig.dataset)
        
        @classmethod
        def to_str(cls): # __str__, self
            dic = cls.__dict__.copy()
            lst = list(filter( \
                            lambda p: (not p[0].startswith('__')) and type(p[1]) != classmethod, \
                            dic.items() \
                            ))
            return '\n'.join([str(k) + ' = ' + str(v) for k, v in lst])
        
    
    class t2vec:
        model = 't2vec'
        
        PAD = 0
        UNK = 3

        PAD_WORD = '<blank>'
        BOS_WORD = '<s>'
        EOS_WORD = '</s>'
        UNK_WORD = '<unk>'
        
        BATCH_SIZE = 16
        grid_num = 50
        vocab_size = grid_num * grid_num + 2
        BOS = grid_num * grid_num + 1
        EOS = grid_num * grid_num
        dropout = 0.1
        embedding_dim = 64
        hidden_dim = 16
        num_layers = 3
        
        learning_rate = 1e-3
        MAX_EPOCH = 300
        max_grad_norm = 5.0
        training_bad_patience = 10
        
        checkpoint_dir = 'exp/{}/t2vec'.format(DatasetConfig.dataset)
        index_dir = root_dir + '/exp/{}/t2vec/index'.format(DatasetConfig.dataset)
        
        @classmethod
        def to_str(cls): # __str__, self
            dic = cls.__dict__.copy()
            lst = list(filter( \
                            lambda p: (not p[0].startswith('__')) and type(p[1]) != classmethod, \
                            dic.items() \
                            ))
            return '\n'.join([str(k) + ' = ' + str(v) for k, v in lst])
        

    class EDR:
        checkpoint_dir = 'exp/{}/EDR'.format(DatasetConfig.dataset)
        
        @classmethod
        def to_str(cls):
            dic = cls.__dict__.copy()
            lst = list(filter( \
                            lambda p: (not p[0].startswith('__')) and type(p[1]) != classmethod, \
                            dic.items() \
                            ))
            return '\n'.join([str(k) + ' = ' + str(v) for k, v in lst])
        
    class EDwP:
        checkpoint_dir = 'exp/{}/EDwP'.format(DatasetConfig.dataset)
        
        @classmethod
        def to_str(cls):
            dic = cls.__dict__.copy()
            lst = list(filter( \
                            lambda p: (not p[0].startswith('__')) and type(p[1]) != classmethod, \
                            dic.items() \
                            ))
            return '\n'.join([str(k) + ' = ' + str(v) for k, v in lst])
        
        