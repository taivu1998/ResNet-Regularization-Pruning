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

from resnet import ResNet
from regularization import *

import warnings
warnings.filterwarnings('ignore')


class Net(pl.LightningModule):
    def __init__(self, arch, criterion, args):
        super(Net, self).__init__()
        self.args = args
        if self.args.seed != 0:
            torch.manual_seed(self.args.seed)

        num_classes = 100 if self.args.dataset == 'cifar100' else 10
        self.net = ResNet(arch, num_classes)
        self.criterion = criterion
        
        self.mixup = Mixup() if self.args.regularize == 'mixup' else None
        # self.cutout = Cutout(n_holes = self.args.n_holes_cutout, length = self.args.length.cutout) \
        #               if self.args.regularize == 'cutout' else None

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
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

    def validation_step(self, batch, batch_nb):
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
        # if self.on_gpu:
        #     acc = acc.cuda(loss.device.index)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'].float() for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'].float() for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr = self.args.lr,
                              momentum = self.args.momentum,
                              weight_decay = self.args.decay)
        # optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum,
        #              weight_decay=self.args.decay)
        if self.args.dataset == 'svhn':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [80, 120], gamma = 0.1,
                                                       last_epoch = self.args.start_epoch-1)
        elif self.args.regularize == 'mixup':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [100, 150], gamma = 0.1,
                                                       last_epoch = self.args.start_epoch-1)
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [60, 120, 160], gamma = 0.2,
                                                       last_epoch=self.args.start_epoch-1)
        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
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
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataloader = self.load_dataset(dataset = self.args.dataset, train = False,
                                       transform = transform_test, shuffle = False,
                                       batch_size = 100)
        return dataloader

    def load_dataset(self, dataset, train, transform, shuffle, batch_size):
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
