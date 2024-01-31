from baseFuncs import constructTrainingData, get_logger, plot_loss
from t2vec import EncoderDecoder
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import constants
from tqdm import trange
import os

BATCH_SIZE = 16
grid_num = 50
vocab_size = grid_num * grid_num + 2
dropout = 0.1
learning_rate = 1e-3
embedding_dim = 64
hidden_dim = 16
MAX_EPOCH = 300
max_grad_norm = 5.0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def NLLcriterion(vocab_size):
    "construct NLL criterion"
    weight = torch.ones(vocab_size)
    weight[constants.PAD] = 0
    ## The first dimension is not batch, thus we need
    ## to average over the batch manually
    #criterion = nn.NLLLoss(weight, size_average=False)
    criterion = nn.NLLLoss(weight, reduction='sum')
    return criterion


def genLoss(m0, m1, train_loader, lossF):
    loss = 0
    for _, x in enumerate(train_loader):
        x = x[0].transpose(0,1).to(device)
        eos = torch.full((1, x.shape[1]), grid_num*grid_num).to(device) # eos = 2500
        sos = torch.full((1, x.shape[1]), grid_num*grid_num+1).to(device) # sos = 2501
        src = torch.cat([x, eos], dim=0)
        tgt = torch.cat([sos, x], dim=0)
        #src (src_seq_len, batch): source tensor
        #lengths (1, batch): source sequence lengths
        src = src.long()
        lengths = torch.full((1, x.shape[1]), x.shape[0]).to(device)
        tgt = tgt.long()
        output = m0(src, lengths, tgt) # (seq_len, batch, hidden_size)
        batch = output.size(1) #16
        ## we want to decode target in range [BOS+1:EOS]
        target = tgt[1:]
        ## (seq_len, generator_batch, hidden_size) =>
        ## (seq_len*generator_batch, hidden_size)
        o = output.view(-1, output.size(2)) # (seq_len*generator_batch, hidden_size)
        o = m1(o) 
        ## (seq_len*generator_batch,)
        t = target.view(-1)
        loss += lossF(o, t)
    return loss.div(batch)


def trainModel(trainFilePath, m0SavePath, m1SavePath, trainlogPath):
    print("Loading data...")
    x = constructTrainingData(trainFilePath, BATCH_SIZE)
    dataSet = torch.utils.data.TensorDataset(torch.from_numpy(x))
    val_size = int(0.2 * len(dataSet))
    
    criterion = NLLcriterion(vocab_size).to(device)
    lossF = lambda o, t: criterion(o, t)
    
    m0 = EncoderDecoder(vocab_size,
                        embedding_dim,
                        hidden_dim,
                        num_layers=3,
                        dropout=dropout,
                        bidirectional=True).to(device)
    m1 = nn.Sequential(nn.Linear(hidden_dim, vocab_size),
                       nn.LogSoftmax(dim=1)).to(device)
        
    m0_optimizer = torch.optim.Adam(m0.parameters(), learning_rate)
    m1_optimizer = torch.optim.Adam(m1.parameters(), learning_rate)
    
    logger = get_logger(trainlogPath)
    train_loss_list = []
    test_loss_list = []
    
    print("Start training...")
    for epoch in trange(MAX_EPOCH):
        train_dataset, _ = torch.utils.data.random_split(dataSet, [len(dataSet) - val_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
        m0_optimizer.zero_grad()
        m1_optimizer.zero_grad()
        m0.train()
        m1.train()
        loss = genLoss(m0, m1, train_loader, lossF)
        loss.backward()
        ## clip the gradients
        clip_grad_norm_(m0.parameters(), max_grad_norm)
        clip_grad_norm_(m1.parameters(), max_grad_norm)
        ## one step optimization
        m0_optimizer.step()
        m1_optimizer.step()
        logger.info('Epoch:[{}/{}]\t Train Loss={:.4f}'.format(epoch+1 , MAX_EPOCH, loss))
        train_loss_list.append(loss)
    print("End training...")
    torch.save(m0, m0SavePath)
    torch.save(m1, m1SavePath)
    args = {'MODEL': 't2vec', 'SSM_KNN': False}
    plot_loss(train_loss_list, test_loss_list, args)
    
    
if __name__ == '__main__':
    root = '../results/t2vec/'
    if not os.path.exists(root):
        os.makedirs(root)
    dataset = ['beijing', 'MCD']
    trainFilePath = '../data/{}/Train/trainGridData/'.format(dataset[0])
    m0SavePath = '../results/t2vec/m0.pt'
    m1SavePath = '../results/t2vec/m1.pt'
    trainlogPath = '../results/t2vec/train.log'
    trainModel(trainFilePath, m0SavePath, m1SavePath, trainlogPath)
    