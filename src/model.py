'''
This program implements a Lightning wrapper for ResNet and trains the model.
'''

from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import pytorch_lightning as pl
from pytorch_lightning import Trainer
import wandb

from resnet import ResNet
from regularization import *
from soft_filter_pruning import Mask

import warnings
warnings.filterwarnings('ignore')


class Net(pl.LightningModule):
    ''' Lightning wrapper for a ResNet model. '''
    
    def __init__(self, arch, criterion, args):
        ''' Initializes the model. '''
        super(Net, self).__init__()
        self.args = args
        if self.args.seed != 0:
            torch.manual_seed(self.args.seed)

        num_classes = 100 if self.args.dataset == 'cifar100' else 10
        self.net = ResNet(arch, num_classes)
        self.criterion = criterion
        self.mixup = Mixup() if self.args.regularize == 'mixup' else None
        
        if self.args.prune == 'soft_filter':
            self.mask = Mask(self.net, self.args)
            self.mask.init_length()
            self.mask.model = self.net
            self.mask.init_mask(self.args.pruning_rate)
            self.mask.do_mask()
            self.net = self.mask.model
        else:
            self.mask = None
            
    def forward(self, x):
        ''' Performs a forward pass through the network. '''
        return self.net(x)

    def training_step(self, batch, batch_nb):
        ''' Trains the model on a batch. '''
        inputs, targets = batch
        if self.args.regularize == 'mixup':
            inputs, targets_a, targets_b, lam = self.mixup.mixup_data(inputs, targets, self.args.alpha_mixup)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
            outputs = self.forward(inputs)
            loss = self.mixup.mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = self.forward(inputs)
            loss = self.criterion(outputs, targets)
        
        tensorboard_logs = {'train_loss': loss}
        _, predicted = torch.max(outputs.data, 1)
        return {'loss': loss, 'progress_bar': tensorboard_logs, 'log': tensorboard_logs}

    def on_epoch_end(self):
        ''' Prunes the network at the end of each epoch. '''
        if self.args.prune == 'soft_filter':
            if self.current_epoch % self.args.epoch_prune == 0 or self.current_epoch == self.args.epochs - 1:
                self.mask.model = self.net
                self.mask.if_zero()
                self.mask.init_mask(self.args.pruning_rate)
                self.mask.do_mask()
                self.mask.if_zero()
                self.net = self.mask.model

    def validation_step(self, batch, batch_nb):
        ''' Evaluates the model on a batch. '''
        inputs, targets = batch
        if self.args.regularize == 'mixup':
            inputs, targets = Variable(inputs, volatile = True), Variable(targets)
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, targets)

        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(targets.data).cpu().sum()
        total = targets.size(0)
        acc = 1.0 * correct / total

        acc = torch.tensor(acc)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_end(self, outputs):
        ''' Records validation outcomes. '''
        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss_mean += output['val_loss']
            val_acc_mean += output['val_acc']
        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        
        tensorboard_logs = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        wandb.log({"Test Accuracy": val_acc_mean, "Test Loss": val_loss_mean})
        return {'avg_val_loss': val_loss_mean, 'progress_bar': tensorboard_logs, 'log': tensorboard_logs}

    def configure_optimizers(self):
        ''' Configures optimizers and learning schedules. '''
        optimizer = optim.SGD(self.parameters(), lr = self.args.lr,
                              momentum = self.args.momentum,
                              weight_decay = self.args.decay)
        if self.args.dataset == 'svhn':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [80, 120], gamma = 0.1,
                                                       last_epoch = self.args.start_epoch - 1)
        elif self.args.regularize == 'cutout':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [60, 120, 160], gamma = 0.2,
                                                       last_epoch=self.args.start_epoch - 1)
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [100, 150], gamma = 0.1,
                                                       last_epoch = self.args.start_epoch - 1)
        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        ''' Loads training dataset. '''
        if self.args.augment:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding = 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])

        if self.args.regularize == 'cutout':
            transform_train.transforms.append(Cutout(n_holes = self.args.n_holes_cutout,
                                                     length = self.args.length_cutout))
            
        dataloader = self.load_dataset(dataset = self.args.dataset, train = True,
                                       transform = transform_train, shuffle = True,
                                       batch_size = self.args.batch_size)
        return dataloader

    @pl.data_loader
    def val_dataloader(self):
        ''' Loads validation dataset. '''
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        dataloader = self.load_dataset(dataset = self.args.dataset, train = False,
                                       transform = transform_test, shuffle = False,
                                       batch_size = 100)
        return dataloader

    @pl.data_loader
    def test_dataloader(self):
        ''' Loads test dataset. '''
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataloader = self.load_dataset(dataset = self.args.dataset, train = False,
                                       transform = transform_test, shuffle = False,
                                       batch_size = 100)
        return dataloader

    def load_dataset(self, dataset, train, transform, shuffle, batch_size):
        ''' Loads a dataset. '''
        if self.args.dataset == 'cifar10':
            dataset = datasets.CIFAR10(root = '~/data',
                                       train = train,
                                       transform = transform,
                                       download = True)
            
        elif self.args.dataset == 'cifar100':
            dataset = datasets.CIFAR100(root = '~/data',
                                        train = train,
                                        transform = transform,
                                        download = True)
        
        elif self.args.dataset == 'stl10':
            split = 'train' if train else 'test'
            dataset = datasets.STL10(root = '~/data',
                                     split=split,
                                     transform = transform,
                                     download = True)

        elif self.args.dataset == 'svhn':
            if train:
                dataset = datasets.SVHN(root = '~/data',
                                        split = 'train',
                                        transform = transform,
                                        download = True)
                extra_dataset = datasets.SVHN(root = '~/data',
                                              split = 'extra',
                                              transform = transform,
                                              download = True)
                # Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
                data = np.concatenate([dataset.data, extra_dataset.data], axis = 0)
                labels = np.concatenate([dataset.labels, extra_dataset.labels], axis = 0)
                dataset.data = data
                dataset.labels = labels
            else:
                dataset = datasets.SVHN(root = '~/data',
                                        split = 'test',
                                        transform = transform,
                                        download = True)
        
        dataloader = torch.utils.data.DataLoader(dataset = dataset,
                                                 batch_size = batch_size,
                                                 shuffle = shuffle,
                                                 pin_memory = True,
                                                 num_workers = 2)
        return dataloader
