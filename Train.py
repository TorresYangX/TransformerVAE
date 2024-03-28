from baseFuncs import *
import torch
import torch.optim as optim
import os
from tqdm import trange
from model.NVAE import TransformerNvib


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

grid_num = 50
Batch_size = 16 #B
dropout = 0.1
learning_rate = 0.001
MAX_EPOCH = 30
ACCUMULATION_STEPS = 1

class Trainer:
    def __init__(self, model, optimizer, train_loader, test_loader, trajectory_length, grid_num, epoch, ACCUMULATION_STEPS):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.trajectory_length = trajectory_length
        self.grid_num = grid_num
        self.epoch = epoch
        self.ACCUMULATION_STEPS = ACCUMULATION_STEPS
    
    def training(self):
        train_losses_value = 0
        for idx, x in enumerate(self.train_loader):
            x = x[0].transpose(0,1).to(device)
            eos = torch.full((1, x.shape[1]), self.grid_num*self.grid_num).to(device)
            sos = torch.full((1, x.shape[1]), self.grid_num*self.grid_num+1).to(device)
            src = torch.cat([x, eos], dim=0)
            tgt = torch.cat([sos, x], dim=0)

            src_key_padding_mask = torch.zeros((x.shape[1], self.trajectory_length + 1), dtype=torch.bool).to(device)
            tgt_key_padding_mask = torch.zeros((x.shape[1], self.trajectory_length + 1), dtype=torch.bool).to(device)

            train_outputs_dict = self.model(
                src,
                tgt,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            train_losses = self.model.loss(**train_outputs_dict, targets=tgt, epoch=self.epoch)
            (train_losses["Loss"] / self.ACCUMULATION_STEPS).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            if ((idx + 1) % self.ACCUMULATION_STEPS == 0) or (idx + 1 == len(self.train_loader)):
                self.optimizer.step()
                self.optimizer.zero_grad()
            train_losses_value += train_losses["Loss"].item()
        return train_losses_value / len(self.train_loader.dataset)

    def evaluation(self):
        test_losses_value = 0
        for _, x in enumerate(self.test_loader):
            with torch.no_grad():
                x = x[0].transpose(0,1).to(device)
                eos = torch.full((1, x.shape[1]), self.grid_num*self.grid_num).to(device)
                sos = torch.full((1, x.shape[1]), self.grid_num*self.grid_num+1).to(device)
                src = torch.cat([x, eos], dim=0)
                tgt = torch.cat([sos, x], dim=0)

                src_key_padding_mask = torch.zeros((x.shape[1], self.trajectory_length + 1), dtype=torch.bool).to(device)
                tgt_key_padding_mask = torch.zeros((x.shape[1], self.trajectory_length + 1), dtype=torch.bool).to(device)

                test_outputs_dict = self.model(
                    src,
                    tgt,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                )
                test_losses = self.model.loss(**test_outputs_dict, targets=tgt, epoch=self.epoch)
                test_losses_value += test_losses["Loss"].item()
        return test_losses_value / len(self.test_loader.dataset)
    

def trainModel(trainFilePath, modelSavePath, trainlogPath, trajectory_length):
    x = constructTrainingData(trainFilePath, Batch_size)
    dataSet = torch.utils.data.TensorDataset(torch.from_numpy(x))
    val_size = int(0.2 * len(dataSet))
    model = TransformerNvib().to(device)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    logger = get_logger(trainlogPath)
    train_loss_list = []
    test_loss_list = []
    print("Start training...")
    for epoch in trange(MAX_EPOCH):
        train_dataset, test_dataset = torch.utils.data.random_split(dataSet, [len(dataSet) - val_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Batch_size)
        trainer = Trainer(model, optimizer, train_loader, test_loader, trajectory_length, grid_num, epoch, ACCUMULATION_STEPS)
        train_loss = trainer.training()
        test_loss = trainer.evaluation()
        if epoch == 0 or train_loss < min(train_loss_list):
            torch.save(model, modelSavePath)
        logger.info('Epoch:[{}/{}]\t Train Loss={:.4f}\t Test Loss={:.4f}'.format(epoch+1 , MAX_EPOCH, train_loss, test_loss ))
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
    print("End training...")
    plot_loss(train_loss_list, test_loss_list)
