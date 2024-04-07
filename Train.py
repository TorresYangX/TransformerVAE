import sys
import logging
import argparse
from model_config import ModelConfig
from model.NVAE_trainer import Trainer
from dataset_config import DatasetConfig
from baseline.DeepModels.AE import AE_Trainer
from baseline.t2vec.t2vec import t2vec_Trainer
from baseline.DeepModels.VAE import VAE_Trainer
from baseline.DeepModels.transformer import Transformer_Trainer


logging.getLogger().setLevel(logging.INFO)

def parse_args():
    # dont set default value here! -- it will incorrectly overwrite the values in config.py.
    # config.py is the correct place for default values.
    
    parser = argparse.ArgumentParser(description = "NVAE/train.py")
    parser.add_argument('--model', type = str, help = '')
    parser.add_argument('--dataset', type = str, help = '')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    DatasetConfig.update(dict(filter(lambda kv: kv[1] is not None, vars(args).items())))
    
    logging.info('python ' + ' '.join(sys.argv))
    logging.info('=================================')
    logging.info(DatasetConfig.to_str())
    logging.info('=================================')
    model_mapping = {
        'NVAE': {'trainer': Trainer, 'config': ModelConfig.NVAE},
        'AE': {'trainer': AE_Trainer, 'config': ModelConfig.AE},
        'VAE': {'trainer': VAE_Trainer, 'config': ModelConfig.VAE},
        'Transformer': {'trainer': Transformer_Trainer, 'config': ModelConfig.Transformer},
        't2vec': {'trainer': t2vec_Trainer, 'config': ModelConfig.t2vec}
    }
    if args.model not in model_mapping:
        raise ValueError('model not found')
    trainer_class = model_mapping[args.model]['trainer']
    config_class = model_mapping[args.model]['config']
    
    logging.info(config_class.to_str())
    logging.info('=================================')

    trainer = trainer_class()
    trainer.train()