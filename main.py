'''

'''

import os, sys
sys.path.append('.')
sys.path.append('./src')
import argparse

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from model import Net

import warnings
warnings.filterwarnings('ignore')


arch_options = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202',
                'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
dataset_options = ['cifar10', 'cifar100', 'svhn', 'imagenet']
regularize_options = [None, 'mixup', 'cutout']
prune_options = [None, 'soft_filter']


def parseArgs():
    ''' Read command line arguments. '''
    parser = argparse.ArgumentParser(description = 'PyTorch ResNet Training.',
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--arch', type = str, default = 'resnet20', help = 'ResNet architecture.', choices = arch_options)
    parser.add_argument('--dataset', type = str, default = 'cifar10', help = 'Dataset.', choices = dataset_options)
    parser.add_argument('--regularize', type = str, default = None, help = 'Regularization.', choices = regularize_options)
    parser.add_argument('--prune', type = str, default = None, help = 'Pruning.', choices = prune_options)
    
    parser.add_argument('--batch-size', type = int, default = 128, help = 'Batch size.')
    parser.add_argument('--lr', type = float, default = 0.1, help = 'Learning rate.')
    parser.add_argument('--start-epoch', type = int, default = 0, help = 'Starting epoch.')
    parser.add_argument('--epochs', type = int, default = 200, help = 'Number of epochs.')
    parser.add_argument('--augment', action = 'store_true', default = False, help = 'Augment data by flipping and cropping.')
    parser.add_argument('--decay', type = float, default = 1e-4, help = 'Weight decay.')
    parser.add_argument('--momentum', default = 0.9, type = float, metavar = 'M', help = 'Momentum.')
    parser.add_argument('--seed', type = int, default = 0, help = 'Random seed.')
    parser.add_argument('--resume', action = 'store_true', default = False, help = 'Resume from checkpoint.')
    
    parser.add_argument('--alpha-mixup', type = float, default = 1., help = 'Mixup interpolation coefficient.')
    parser.add_argument('--n-holes-cutout', type = int, default = 1, help = 'Number of holes to cut out from image.')
    parser.add_argument('--length-cutout', type = int, default = 16, help = 'Length of the holes.')
    
    args = parser.parse_known_args()[0]
    return args


def main():
    ''' Main program. '''
    print("Welcome to Our CNN Program.")
    args = parseArgs()
    
    model = Net(arch = args.arch, criterion = nn.CrossEntropyLoss(), args = args)
    gpus = 1 if torch.cuda.is_available() else 0
    trainer = Trainer(gpus = gpus, min_nb_epochs = 1, max_nb_epochs = args.epochs,
                      early_stop_callback = None, check_val_every_n_epoch = 1)
    
    trainer.fit(model)
    print('View tensorboard logs by running\ntensorboard --logdir %s' % os.getcwd())
    print('and going to http://localhost:6006 on your browser')
    trainer.test()
    
    
if __name__ == '__main__':
    main()
