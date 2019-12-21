'''
This program calculates the number of floating point operations (FLOPs)
for various network architectures and pruning rates.

References:
    https://github.com/he-y/soft-filter-pruning
'''

def cifar_resnet_flop(layer = 110, prune_rate = 1):
    '''
    Compares the number of FLOPs for a ResNet model with a pruning rate.
    
    Args:
        layer (int): The ResNet network size for CIFAR.
        prune_rate (int): Compression rate, 1 means baseline.
        
    Returns:
        int: The number of FLOPs of the network.
    '''
    flop = 0
    channel = [16, 32, 64]
    width = [32, 16, 8]

    stage = int(layer / 3)
    for index in range(0, layer, 1):
        if index == 0:  # first conv layer before block
            flop += channel[0] * width[0] * width[0] * 9 * 3 * prune_rate
        elif index in [1, 2]:  # first block of first stage
            flop += channel[0] * width[0] * width[0] * 9 * channel[0] * (prune_rate ** 2)
        elif 2 < index <= stage:  # other blocks of first stage
            if index % 2 != 0:
                # first layer of block, only output channal reduced, input channel remain the same
                flop += channel[0] * width[0] * width[0] * 9 * channel[0] * (prune_rate)
            elif index % 2 == 0:
                # second layer of block, both input and output channal reduced
                flop += channel[0] * width[0] * width[0] * 9 * channel[0] * (prune_rate ** 2)
        elif stage < index <= stage * 2:  # second stage
            if index % 2 != 0:
                flop += channel[1] * width[1] * width[1] * 9 * channel[1] * (prune_rate)
            elif index % 2 == 0:
                flop += channel[1] * width[1] * width[1] * 9 * channel[1] * (prune_rate ** 2)
        elif stage * 2 < index <= stage * 3:  # third stage
            if index % 2 != 0:
                flop += channel[2] * width[2] * width[2] * 9 * channel[2] * (prune_rate)
            elif index % 2 == 0:
                flop += channel[2] * width[2] * width[2] * 9 * channel[2] * (prune_rate ** 2)

    # offset for dimension change between blocks
    offset1 = channel[1] * width[1] * width[1] * 9 * channel[1] * prune_rate - channel[1] * width[1] * width[1] * 9 * \
              channel[0] * prune_rate
    offset2 = channel[2] * width[2] * width[2] * 9 * channel[2] * prune_rate - channel[2] * width[2] * width[2] * 9 * \
              channel[1] * prune_rate
    flop = flop - offset1 - offset2
    return flop


def cal_cifar_resnet_flop(layer, prune_rate):
    '''
    Compares the number of FLOPs for a ResNet model
    with and  without a pruning rate.
    
    Args:
        layer (int): The ResNet network size for CIFAR.
        prune_rate (int): Compression rate, 1 means baseline.
    '''
    pruned_flop = cifar_resnet_flop(layer, prune_rate)
    baseline_flop = cifar_resnet_flop(layer, 1)

    print(
        "pruning rate of layer {:d} is {:.1f}, pruned FLOP is {:.0f}, "
        "baseline FLOP is {:.0f}, FLOP reduction rate is {:.4f}"
        .format(layer, prune_rate, pruned_flop, baseline_flop, 1 - pruned_flop / baseline_flop))


def main():
    ''' Main program. '''
    layer_list = [20, 32, 44, 56, 110, 1202,
                  18, 34, 50, 101, 152]
    pruning_rate_list = [0.9, 0.8, 0.7]
    for layer in layer_list:
        for pruning_rate in pruning_rate_list:
            cal_cifar_resnet_flop(layer, pruning_rate)


if __name__ == '__main__':
    main()
