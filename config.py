class Config:
    
    # ================== NVAE ==================
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
    MAX_EPOCH = 500
    ACCUMULATION_STEPS = 1