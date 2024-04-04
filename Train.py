# from baseFuncs import *
# import torch
# import torch.optim as optim
# import os
# from tqdm import trange
# from model.NVAE import TransformerNvib


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# grid_num = 50
# Batch_size = 16 #B
# dropout = 0.1
# learning_rate = 0.001
# MAX_EPOCH = 30
# ACCUMULATION_STEPS = 1
    

# def trainModel(trainFilePath, modelSavePath, trainlogPath, trajectory_length):
#     x = constructTrainingData(trainFilePath, Batch_size)
#     dataSet = torch.utils.data.TensorDataset(torch.from_numpy(x))
#     val_size = int(0.2 * len(dataSet))
#     model = TransformerNvib().to(device)
#     optimizer = optim.Adam(model.parameters(),lr=learning_rate)
#     logger = get_logger(trainlogPath)
#     train_loss_list = []
#     test_loss_list = []
#     print("Start training...")
#     for epoch in trange(MAX_EPOCH):
#         train_dataset, test_dataset = torch.utils.data.random_split(dataSet, [len(dataSet) - val_size, val_size])
#         train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size)
#         test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Batch_size)
#         trainer = Trainer(model, optimizer, train_loader, test_loader, trajectory_length, grid_num, epoch, ACCUMULATION_STEPS)
#         train_loss = trainer.training()
#         test_loss = trainer.evaluation()
#         if epoch == 0 or train_loss < min(train_loss_list):
#             torch.save(model, modelSavePath)
#         logger.info('Epoch:[{}/{}]\t Train Loss={:.4f}\t Test Loss={:.4f}'.format(epoch+1 , MAX_EPOCH, train_loss, test_loss ))
#         train_loss_list.append(train_loss)
#         test_loss_list.append(test_loss)
#     print("End training...")
#     plot_loss(train_loss_list, test_loss_list)

import argparse
from model_config import ModelConfig
from dataset_config import DatasetConfig
from model.NVAE_trainer import Trainer

def parse_args():
    # dont set default value here! -- it will incorrectly overwrite the values in config.py.
    # config.py is the correct place for default values.
    
    parser = argparse.ArgumentParser(description = "NVAE/train.py")
    parser.add_argument('--dataset', type = str, help = '')

    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))

if __name__ == '__main__':
    DatasetConfig.update(parse_args())
    trainer = Trainer()
    # trainer.train()
    trainer.encode('test')
