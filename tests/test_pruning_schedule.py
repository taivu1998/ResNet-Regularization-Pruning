import os
import sys
import types
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(REPO_ROOT, 'src')

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


if 'pytorch_lightning' not in sys.modules:
    pytorch_lightning = types.ModuleType('pytorch_lightning')

    class LightningModule(nn.Module):
        def __init__(self):
            super(LightningModule, self).__init__()
            self.current_epoch = 0

    def data_loader(fn):
        return fn

    pytorch_lightning.LightningModule = LightningModule
    pytorch_lightning.Trainer = object
    pytorch_lightning.data_loader = data_loader
    sys.modules['pytorch_lightning'] = pytorch_lightning


if 'wandb' not in sys.modules:
    wandb = types.ModuleType('wandb')
    wandb.init = lambda *args, **kwargs: None
    wandb.watch = lambda *args, **kwargs: None
    wandb.log = lambda *args, **kwargs: None
    wandb.save = lambda *args, **kwargs: None
    sys.modules['wandb'] = wandb


from resnet import ResNet
from model import Net


def build_args(**overrides):
    args = {
        'arch': 'resnet20',
        'dataset': 'cifar10',
        'regularize': None,
        'prune': 'soft_filter',
        'seed': 123,
        'pruning_rate': 0.9,
        'epoch_prune': 1,
        'prune_start_epoch': 1,
        'epochs': 5,
        'alpha_mixup': 1.0,
        'n_holes_cutout': 1,
        'length_cutout': 16,
        'augment': False,
        'batch_size': 128,
        'lr': 0.1,
        'momentum': 0.9,
        'decay': 1e-4,
        'start_epoch': 0,
        'skip_downsample': 1,
        'layer_begin': 0,
        'layer_end': 54,
        'layer_inter': 3,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


class PruningScheduleTest(unittest.TestCase):
    def test_constructor_does_not_apply_soft_pruning(self):
        args = build_args()

        torch.manual_seed(args.seed)
        reference_model = ResNet('resnet20', 10)

        net = Net(arch = 'resnet20', criterion = nn.CrossEntropyLoss(), args = args)

        for reference_param, actual_param in zip(reference_model.parameters(), net.net.parameters()):
            self.assertTrue(torch.equal(reference_param, actual_param))

    def test_pruning_respects_prune_start_epoch(self):
        args = build_args(prune_start_epoch = 2)
        net = Net(arch = 'resnet20', criterion = nn.CrossEntropyLoss(), args = args)

        original_weight = net.net.conv1.weight.detach().clone()

        net.current_epoch = 0
        net.on_epoch_end()
        self.assertTrue(torch.equal(original_weight, net.net.conv1.weight))

        net.current_epoch = 1
        net.on_epoch_end()

        pruned_weight = net.net.conv1.weight.detach().clone()
        self.assertFalse(torch.equal(original_weight, pruned_weight))

        filters = pruned_weight.view(pruned_weight.size(0), -1)
        zeroed_filters = (filters.abs().sum(dim = 1) == 0).sum().item()
        self.assertGreater(zeroed_filters, 0)


if __name__ == '__main__':
    unittest.main()
