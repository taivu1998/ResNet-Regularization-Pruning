# How Not to Give a FLOP: Combining Regularization and Pruning for Efficient Inference

[![Paper](https://img.shields.io/badge/arXiv-2003.13593-b31b1b.svg)](https://arxiv.org/abs/2003.13593)
[![Framework](https://img.shields.io/badge/PyTorch-Lightning-792ee5.svg)](https://www.pytorchlightning.ai/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

A research framework for **model compression** via the joint application of **structured filter pruning** and **regularization-induced sparsity** on deep residual networks. We demonstrate that combining soft filter pruning with Mixup/Cutout regularization yields a **~15% reduction in FLOPs** with up to a **+2% improvement in top-1 accuracy** over unpruned baselines on CIFAR-10, establishing that these techniques are complementary rather than competing.

[[Paper]](https://arxiv.org/abs/2003.13593) | [[Code]](https://github.com/taivu1998/ResNet-Regularization-Pruning)

---

## Table of Contents

- [Key Results](#key-results)
- [Motivation](#motivation)
- [Technical Approach](#technical-approach)
  - [Structured Filter Pruning via L2-Norm Ranking](#1-structured-filter-pruning-via-l2-norm-ranking)
  - [Mixup Regularization](#2-mixup-regularization)
  - [Cutout Regularization](#3-cutout-regularization)
  - [Joint Regularization-Pruning Pipeline](#4-joint-regularization-pruning-pipeline)
- [Architecture](#architecture)
- [Experimental Results](#experimental-results)
- [Analysis](#analysis)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [References](#references)

---

## Key Results

Evaluated on **CIFAR-10** across five ResNet depths (20/32/44/56/110), training for 200 epochs with SGD (lr=0.1, momentum=0.9, weight decay=1e-4):

| Method | ResNet-20 | ResNet-32 | ResNet-44 | ResNet-56 | ResNet-110 | Avg. FLOP Reduction |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline (Control) | 91.63% | 92.11% | 92.54% | 92.49% | 92.58% | — |
| + Mixup | 92.67% | 93.39% | 94.16% | 94.15% | 94.71% | 0% |
| + Cutout | 91.81% | 93.48% | 93.60% | 93.86% | 93.95% | 0% |
| + Soft Filter Pruning | 91.47% | 91.40% | 92.23% | 91.42% | 91.88% | **~15%** |
| **+ Mixup & Pruning** | **91.94%** | **93.06%** | **93.12%** | **93.80%** | **93.04%** | **~15%** |
| **+ Cutout & Pruning** | **92.87%** | **93.28%** | **94.12%** | **94.52%** | **94.57%** | **~15%** |

**Headline finding:** Cutout + Pruning on ResNet-56 achieves **+2.03% accuracy** and **14.7% fewer FLOPs** compared to the unpruned baseline — a strict Pareto improvement on both the accuracy and efficiency frontiers.

---

## Motivation

Deep convolutional networks are heavily over-parameterized: empirically, a large fraction of learned filters contribute minimally to the final prediction. While prior work has studied **regularization** (to combat overfitting) and **pruning** (to reduce model size) as independent concerns, few works have examined their joint interaction. We hypothesize that regularization produces weight distributions that are more amenable to magnitude-based pruning by encouraging smoother decision boundaries and more uniformly distributed feature representations. This work investigates and confirms this hypothesis: the combination yields a strict improvement over either technique applied in isolation.

---

## Technical Approach

### 1. Structured Filter Pruning via L2-Norm Ranking

Unlike unstructured (weight-level) pruning that produces irregular sparse matrices requiring specialized hardware, **structured pruning** removes entire convolutional filters, yielding architecturally smaller dense networks that benefit from standard hardware acceleration without custom sparse compute kernels.

We adopt the **Soft Filter Pruning** strategy (He et al., 2018), which avoids the catastrophic accuracy loss of hard one-shot pruning by iteratively zeroing and retraining filters:

**Filter Selection Criterion:** For each convolutional layer with filters $\mathbf{F}_i \in \mathbb{R}^{C_{in} \times k \times k}$, compute the L2-norm:

$$\lVert \mathbf{F}_i \rVert_2 = \sqrt{\sum_{c=1}^{C_{in}} \sum_{p=1}^{k} \sum_{q=1}^{k} \mathbf{F}_i(c, p, q)^2}$$

Filters are ranked by this norm. Given a pruning rate $r \in [0,1]$, the bottom $(1-r) \times N$ filters per layer are zeroed.

**Iterative Soft Pruning Schedule:**

```
for epoch = 1 to T:
    1. Train network for one epoch (backprop updates all weights, including previously zeroed filters)
    2. Recompute L2-norm ranking across all filters
    3. Apply binary mask: zero out lowest-norm filters per layer
    4. Repeat until convergence
```

The key insight is **soft** pruning: zeroed filters are not permanently removed during training. Backpropagation can resurrect previously pruned filters if they become salient, enabling the network to dynamically reallocate capacity. Final hard pruning occurs post-training, removing all zero-valued filters to produce the compressed architecture.

**Downsample-aware pruning:** Shortcut projection layers (1x1 convolutions for dimension matching in residual connections) are excluded from pruning to preserve gradient flow through skip connections.

<p align="center">
  <img src="https://user-images.githubusercontent.com/46636857/77240994-3c51c800-6c1f-11ea-9dad-d970c34aefbb.png">
  <br>
  <em>Figure 1: Soft filter pruning pipeline. Filters are iteratively zeroed and retrained, allowing the network to redistribute representational capacity before permanent removal (He et al., 2018).</em>
</p>

### 2. Mixup Regularization

Mixup (Zhang et al., 2017) trains the network on virtual examples constructed via convex combinations of input pairs, enforcing linear interpolation behavior in the learned feature space:

$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j, \quad \tilde{y} = \lambda y_i + (1 - \lambda) y_j$$

where $\lambda \sim \text{Beta}(\alpha, \alpha)$ and $(x_i, y_i), (x_j, y_j)$ are drawn randomly from the training set.

The modified loss becomes:

$$\mathcal{L}_{\text{mixup}} = \lambda \cdot \mathcal{L}(\hat{y}, y_i) + (1 - \lambda) \cdot \mathcal{L}(\hat{y}, y_j)$$

Mixup acts as a data-dependent regularizer that reduces oscillatory behavior in the loss landscape, producing smoother weight distributions. We hypothesize this smoothness makes the resulting filter norms more separable, improving the discriminability of the L2-norm pruning criterion.

<p align="center">
  <img src="https://user-images.githubusercontent.com/46636857/77240982-26440780-6c1f-11ea-94ea-509ae64bdaab.jpg">
  <br>
  <em>Figure 2: Mixup generates synthetic training samples by linearly interpolating between image pairs and their labels (Zhang et al., 2017).</em>
</p>

### 3. Cutout Regularization

Cutout (DeVries & Taylor, 2017) applies spatial dropout by masking contiguous square regions of the input during training:

$$\tilde{x} = x \odot M, \quad M \in \{0, 1\}^{H \times W}$$

where $M$ is a binary mask with $n$ square holes of side length $l$ at random positions. This forces the network to rely on distributed spatial representations rather than localizing predictions on small discriminative regions, yielding more robust and generalizable features.

<p align="center">
  <img src="https://user-images.githubusercontent.com/46636857/77240985-31973300-6c1f-11ea-9b5e-c88bfbc73b41.jpg">
  <br>
  <em>Figure 3: Cutout applied to CIFAR-10 images. Random square patches are zeroed during training to improve feature robustness (DeVries & Taylor, 2017).</em>
</p>

### 4. Joint Regularization-Pruning Pipeline

The full training pipeline integrates both techniques into a unified optimization loop:

```
Initialize: ResNet model with Kaiming normal initialization
            Binary pruning mask M (all ones initially)

for epoch = 1 to 200:
    for each mini-batch (x, y):
        if Mixup:
            λ ~ Beta(α, α)
            x̃ = λx + (1-λ)x[shuffle]
            loss = λ·CE(f(x̃), y) + (1-λ)·CE(f(x̃), y[shuffle])
        elif Cutout:
            x̃ = x ⊙ RandomMask(n_holes, length)
            loss = CE(f(x̃), y)
        else:
            loss = CE(f(x), y)

        θ ← θ - η·∇loss (SGD with momentum + weight decay)

    # Structured pruning step (epoch boundary)
    for each conv layer l:
        Rank filters by ‖F_i‖₂
        Zero bottom (1-r)% of filters via mask
```

---

## Architecture

We implement the full family of ResNet architectures (He et al., 2015) spanning two design regimes:

### CIFAR-Scale (32x32 Input)

Three-stage residual networks with `BasicBlock` (two 3x3 conv layers per block). Shortcut connections use zero-padding (Option A) for dimension changes.

| Model | Blocks per Stage | Layers | Parameters | MFLOPs |
|:---|:---:|:---:|:---:|:---:|
| ResNet-20 | [3, 3, 3] | 20 | 0.27M | 40.55 |
| ResNet-32 | [5, 5, 5] | 32 | 0.46M | 68.86 |
| ResNet-44 | [7, 7, 7] | 44 | 0.66M | 97.17 |
| ResNet-56 | [9, 9, 9] | 56 | 0.85M | 125.49 |
| ResNet-110 | [18, 18, 18] | 110 | 1.73M | 252.89 |
| ResNet-1202 | [200, 200, 200] | 1202 | 19.4M | 2,813.6 |

Channel progression: `16 → 32 → 64` across stages at spatial resolutions `32² → 16² → 8²`.

### ImageNet-Scale

Four-stage residual networks supporting both `BasicBlock` and `Bottleneck` (1x1-3x3-1x1) configurations with learned 1x1 projection shortcuts.

| Model | Block Type | Blocks per Stage | Layers |
|:---|:---:|:---:|:---:|
| ResNet-18 | Basic | [2, 2, 2, 2] | 18 |
| ResNet-34 | Basic | [3, 4, 6, 3] | 34 |
| ResNet-50 | Bottleneck | [3, 4, 6, 3] | 50 |
| ResNet-101 | Bottleneck | [3, 4, 23, 3] | 101 |
| ResNet-152 | Bottleneck | [3, 8, 36, 3] | 152 |

Channel progression: `64 → 128 → 256 → 512` (×4 expansion for bottleneck blocks).

---

## Experimental Results

### Baseline Accuracy vs. Network Depth (CIFAR-10)

| Model | MFLOPs | Top-1 Accuracy |
|:---|:---:|:---:|
| ResNet-20 | 40.55 | 91.63% |
| ResNet-32 | 68.86 | 92.11% |
| ResNet-44 | 97.17 | 92.54% |
| ResNet-56 | 125.49 | 92.49% |
| ResNet-110 | 252.89 | 92.58% |

Accuracy saturates beyond ResNet-32 (~92.1%), despite a 3.7x increase in FLOPs from ResNet-32 to ResNet-110. This confirms significant over-parameterization and the opportunity for compression.

### Regularization Impact (No Pruning)

| Model | Baseline | + Mixup (Δ) | + Cutout (Δ) |
|:---|:---:|:---:|:---:|
| ResNet-20 | 91.63% | 92.67% (+1.04) | 91.81% (+0.18) |
| ResNet-32 | 92.11% | 93.39% (+1.28) | 93.48% (+1.37) |
| ResNet-44 | 92.54% | 94.16% (+1.62) | 93.60% (+1.06) |
| ResNet-56 | 92.49% | 94.15% (+1.66) | 93.86% (+1.37) |
| ResNet-110 | 92.58% | 94.71% (+2.13) | 93.95% (+1.37) |

Regularization improves accuracy without affecting FLOPs (no architectural change at inference). Mixup benefits scale with network depth, suggesting deeper networks have more latent capacity that regularization helps exploit.

### Pruning Impact (No Regularization)

| Model | Baseline Acc. | Pruned Acc. (Δ) | FLOP Reduction |
|:---|:---:|:---:|:---:|
| ResNet-20 | 91.63% | 91.47% (−0.16) | 15.2% |
| ResNet-32 | 92.11% | 91.40% (−0.71) | 14.9% |
| ResNet-44 | 92.54% | 92.23% (−0.31) | 14.8% |
| ResNet-56 | 92.49% | 91.42% (−1.07) | 14.7% |
| ResNet-110 | 92.58% | 91.88% (−0.70) | 14.6% |

At a 0.9 pruning rate (10% of filters removed per layer), soft filter pruning achieves a consistent ~15% FLOP reduction with modest accuracy degradation (<1.1% in all cases).

### Combined Regularization + Pruning

| Model | Cutout + Pruning Acc. (Δ vs. Baseline) | Mixup + Pruning Acc. (Δ vs. Baseline) | FLOP Reduction |
|:---|:---:|:---:|:---:|
| ResNet-20 | **92.87%** (+1.24) | 91.94% (+0.31) | 15.2% |
| ResNet-32 | **93.28%** (+1.17) | 93.06% (+0.95) | 14.9% |
| ResNet-44 | **94.12%** (+1.58) | 93.12% (+0.58) | 14.8% |
| ResNet-56 | **94.52%** (+2.03) | 93.80% (+1.31) | 14.7% |
| ResNet-110 | **94.57%** (+1.99) | 93.04% (+0.46) | 14.6% |

**Cutout + Pruning is the dominant strategy:** it achieves the best accuracy-efficiency trade-off, simultaneously improving accuracy by up to +2.03% while reducing FLOPs by ~15%. This Pareto-dominant result demonstrates that regularization and structured pruning are not merely compatible — they are synergistic.

---

## Analysis

### Why the Combination Works

1. **Regularization redistributes filter importance.** By encouraging the network to learn more distributed representations (via spatial dropout or interpolated training samples), regularization prevents the concentration of discriminative power in a small subset of filters. This makes the remaining filters after pruning individually more informative.

2. **Pruning acts as implicit regularization.** Removing low-norm filters constrains model capacity, reducing overfitting — particularly beneficial for medium-depth networks (ResNet-44) where baseline accuracy already plateaus.

3. **Soft pruning enables capacity reallocation.** Because zeroed filters can be reactivated by gradient updates, the network can iteratively converge on the optimal subset of filters under the combined regularization + sparsity pressure.

### FLOP Reduction Analysis

Per-layer FLOP computation follows:

$$\text{FLOPs}_l = C_{\text{out}} \times H \times W \times k^2 \times C_{\text{in}} \times r^2$$

where $r$ is the effective pruning rate (fraction of filters retained). With a 0.9 pruning rate applied uniformly across convolutional layers, the quadratic dependence on $r$ (from pruning both input and output channels of consecutive layers) amplifies the per-layer reduction, yielding ~15% total FLOP savings even with only 10% of filters pruned per layer.

---

## Project Structure

```
ResNet-Regularization-Pruning/
├── main.py                       # Training entry point & CLI argument parsing
├── flopsCount.py                 # Analytical FLOP computation per architecture/pruning rate
└── src/
    ├── model.py                  # PyTorch Lightning training loop (train/val/test + pruning hooks)
    ├── resnet.py                 # Full ResNet family (BasicBlock, Bottleneck, Pre-Act variants)
    ├── regularization.py         # Mixup (Beta-distributed interpolation) & Cutout (spatial masking)
    └── soft_filter_pruning.py    # L2-norm filter ranking, mask generation, iterative pruning
```

---

## Installation

### Prerequisites

- Python 3.6+
- CUDA-compatible GPU (recommended)

### Dependencies

```bash
pip install torch torchvision pytorch-lightning wandb numpy
```

### Datasets

CIFAR-10, CIFAR-100, SVHN, and STL-10 are automatically downloaded via `torchvision.datasets` to `~/data/`.

---

## Usage

### Training with Combined Regularization + Pruning

```bash
# Cutout + Soft Filter Pruning on ResNet-56 (best configuration)
python main.py \
    --arch resnet56 \
    --dataset cifar10 \
    --regularize cutout \
    --prune soft_filter \
    --epochs 200 \
    --lr 0.1 \
    --batch-size 128 \
    --pruning-rate 0.9 \
    --n-holes-cutout 1 \
    --length-cutout 16 \
    --augment

# Mixup + Pruning on ResNet-110
python main.py \
    --arch resnet110 \
    --dataset cifar10 \
    --regularize mixup \
    --prune soft_filter \
    --alpha-mixup 1.0 \
    --pruning-rate 0.9

# Baseline (no regularization, no pruning)
python main.py --arch resnet20 --dataset cifar10
```

### Computing FLOPs

```bash
python flopsCount.py
```

Outputs analytical FLOP counts for all ResNet variants at pruning rates {0.9, 0.8, 0.7}.

### Full CLI Options

| Argument | Default | Description |
|:---|:---:|:---|
| `--arch` | `resnet20` | Architecture: `resnet{20,32,44,56,110,1202,18,34,50,101,152}` |
| `--dataset` | `cifar10` | Dataset: `cifar10`, `cifar100`, `svhn`, `stl10`, `imagenet` |
| `--regularize` | `None` | Regularization: `mixup`, `cutout` |
| `--prune` | `None` | Pruning: `soft_filter` |
| `--batch-size` | `128` | Training batch size |
| `--lr` | `0.1` | Initial learning rate |
| `--epochs` | `200` | Number of training epochs |
| `--momentum` | `0.9` | SGD momentum |
| `--decay` | `1e-4` | L2 weight decay |
| `--alpha-mixup` | `1.0` | Mixup Beta distribution parameter α |
| `--n-holes-cutout` | `1` | Number of cutout patches per image |
| `--length-cutout` | `16` | Cutout patch side length (pixels) |
| `--pruning-rate` | `0.9` | Fraction of filters retained per layer |
| `--epoch-prune` | `1` | Pruning frequency (every N epochs) |
| `--prune-start-epoch` | `1` | First completed epoch after which pruning can begin |
| `--skip-downsample` | `1` | Skip pruning on shortcut projection layers |
| `--augment` | `False` | Enable random crop + horizontal flip |
| `--seed` | `0` | Random seed (0 = no seeding) |

### Experiment Tracking

Training metrics (loss, accuracy) are logged to [Weights & Biases](https://wandb.ai) and TensorBoard in real time.

---

## Authors

- **Tai Vu** — Stanford University
- **Emily Wen** — Stanford University
- **Roy Nehoran** — Stanford University

## Citation

```bibtex
@article{vu2020flop,
  title={How Not to Give a FLOP: Combining Regularization and Pruning for Efficient Inference},
  author={Vu, Tai and Wen, Emily and Nehoran, Roy},
  journal={arXiv preprint arXiv:2003.13593},
  year={2020}
}
```

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2015). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). *CVPR 2016*.
2. He, Y., Kang, G., Dong, X., Fu, Y., & Yang, Y. (2018). [Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks](https://arxiv.org/abs/1808.06866). *IJCAI 2018*.
3. Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). [Mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412). *ICLR 2018*.
4. DeVries, T., & Taylor, G. W. (2017). [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552). *arXiv preprint*.
5. Han, S., Pool, J., Tung, J., & Dally, W. J. (2015). [Learning Both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626). *NeurIPS 2015*.
6. Collins, M. D., & Kohli, P. (2014). [Memory Bounded Deep Convolutional Networks](https://arxiv.org/abs/1412.1442). *arXiv preprint*.
