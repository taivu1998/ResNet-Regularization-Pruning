# ResNet-Regularization-Pruning

This program provides a Pytorch Lightning implementation of the ResNet architecture with mixup and cutout regularizations and soft filter pruning. The goal is to reduce the number of floating-point operations (FLOPs) while improving predictive accuracy. [[Paper]](https://arxiv.org/abs/2003.13593)

## Methods

### Mixup Regularization

Mixup adds convex combinations of pairs of examples and their labels to the training data, which improves the generalization error of cutting-edge network architectures, alleviates the memorization of corrupt labels, and increases sensitivity to adversarial examples (Zhang et al., 2017).

<p align="center">
  <img src="https://user-images.githubusercontent.com/46636857/77240982-26440780-6c1f-11ea-94ea-509ae64bdaab.jpg">
  <br>
    <em>Figure 1: Illustration of mixup, from Zhang et al. (2017).</em>
</p>

### Cutout Regulaization

Cutput randomly masks out contiguous square regions of the input images to CNNs, which improves robustness and yields higher accuracy level (DeVries & Taylor, 2017).

<p align="center">
  <img src="https://user-images.githubusercontent.com/46636857/77240985-31973300-6c1f-11ea-9b5e-c88bfbc73b41.jpg">
  <br>
    <em>Figure 2: Cutout applied to images from the CIFAR-10 dataset, from DeVries & Taylor (2017).</em>
</p>

### Soft Filter Pruning

Soft filter pruning simultaneously trains the model and prunes convolutional filters below some threshold in every epoch, which generates a compact model at the end (He et al., 2018).

<p align="center">
  <img src="https://user-images.githubusercontent.com/46636857/77240994-3c51c800-6c1f-11ea-9dad-d970c34aefbb.png">
  <br>
    <em>Figure 3: Steps of soft filter pruning, from He et al. (2018).</em>
</p>

## Installation

The project requires the following frameworks:

- PyTorch: https://pytorch.org

- PyTorch Lightning: https://github.com/williamFalcon/pytorch-lightning

- W&B: https://www.wandb.com

## Usage

To run the program, use the following command:

```bash
python main.py
```

There are several optional command line arguments:

- --arch: ResNet architecture, such as 'resnet20' or 'resnet18'.
- --dataset: Dataset, such as 'cifar10' or 'cifar100'.
- --regularize: Regularization techniques, such as 'mixup' or 'cutout'.
- --prune: Pruning techniques, such as 'soft_filter'.
- --batch-size: Size of a training batch.
- --lr: Learning rate.
- --epochs: Number of training epochs.
- --alpha-mixup: Mixup interpolation coefficient.
- --n-holes-cutout: Number of holes to cut out from image.
- --length-cutout: Length of the holes in cutout.
- --pruning-rate: Compress rate of a model.

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

