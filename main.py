from train import trainModel
from encode import IndexEncoder
import os
import argparse

def main(args):
    trajectory_length = 60
    root = '../results/{}/NVAE/'.format(args.dataset)
    if not os.path.exists(root):
        os.makedirs(root)
    save_model = root+ 'NVAE.pt'
    trainlog = root + 'trainlog.csv'
    if args.dataset == "Geolife":
        trainFilePath = '../data/Geolife/Train/trainGridData/'
        dataPath = '../data/Geolife/Experiment/experimentGridData/'
    else:
        trainFilePath = '../data/Porto/gridData/'
        dataPath = '../data/Porto/gridData/'
    
    if args.task=="train":
        trainModel(trainFilePath, save_model, trainlog, trajectory_length)
    else:
        indexEncoder = IndexEncoder(save_model, trajectory_length, 50, 16, 16)
        indexEncoder.encoding(dataPath, root)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--task", type=str, default="train", choices=["train","encode"] ,help="train or encode", required=True)

    parser.add_argument("-d", "--dataset", type=str, default="Geolife", choices=["Geolife","Porto"] ,help="dataset", required=True)
    
    args = parser.parse_args()

    main(args)