# ResNet-Regularization-Pruning

This program provides a Pytorch Lightning implementation of the ResNet architecture with mixup and cutout regularizations and soft filter pruning. The goal is to reduce the number of FLOPs while improving predictive accuracy.

## Installation

The project requires the following frameworks:

- PyTorch: https://pytorch.org

- PyTorch Lightning: https://github.com/williamFalcon/pytorch-lightning

## Usage

To run the program, use the following command:

```bash
python main.py
```

There are several optional command line arguments:

- arch: ResNet architecture, such as 'resnet20' or 'resnet18'
- dataset: Dataset, such as 'cifar10' or 'cifar100'
- regularize: Regularization techniques, such as mixup or cutout
- prune: Pruning techniques, such as soft filter pruning

To calculate the number of FLOPs, use the following command:

```bash
python flopsCount.py
```

## Authors

* **Tai Vu** - Stanford University
* **Emily Wen** - Stanford University
* **Roy Nehoran** - Stanford University
