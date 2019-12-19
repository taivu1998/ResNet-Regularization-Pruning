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

- --arch: ResNet architecture, such as 'resnet20' or 'resnet18'
- --dataset: Dataset, such as 'cifar10' or 'cifar100'
- --regularize: Regularization techniques, such as 'mixup' or 'cutout'
- --prune: Pruning techniques, such as 'soft_filter'

To calculate the number of FLOPs, use the following command:

```bash
python flopsCount.py
```

## Authors

* **Tai Vu** - Stanford University
* **Emily Wen** - Stanford University
* **Roy Nehoran** - Stanford University

## References

- DeVries, Terrance, and Graham W. Taylor. “Improved Regularization of Convolutional Neural Networks with Cutout.” ArXiv:1708.04552 [Cs], November 29, 2017. http://arxiv.org/abs/1708.04552.
- He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. “Deep Residual Learning for Image Recognition.” ArXiv:1512.03385 [Cs], December 10, 2015. http://arxiv.org/abs/1512.03385.
- He, Yang, Guoliang Kang, Xuanyi Dong, Yanwei Fu, and Yi Yang. “Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks.” ArXiv:1808.06866 [Cs], August 21, 2018. http://arxiv.org/abs/1808.06866.
- Zhang, Hongyi, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. “Mixup: Beyond Empirical Risk Minimization.” ArXiv:1710.09412 [Cs, Stat], April 27, 2018. http://arxiv.org/abs/1710.09412.

