'''
This program implements soft filter pruning for ResNet.

References:
    https://github.com/he-y/soft-filter-pruning
'''

import torch
import numpy as np


class Mask(object):
    ''' A Mask that performs soft filter pruning. '''

    def __init__(self, model, args):
        ''' Initializes the mask. '''
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.mat = {}
        self.mask_index = []
        self.device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.args = args

    def get_codebook(self, weight_torch, compress_rate, length):
        ''' Gets codebook. '''
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()

        weight_abs = np.abs(weight_np)
        weight_sort = np.sort(weight_abs)

        threshold = weight_sort[int(length * (1 - compress_rate))]
        weight_np[weight_np <= -threshold] = 1
        weight_np[weight_np >= threshold] = 1
        weight_np[weight_np != 1] = 0

        print("Codebook done.")
        return weight_np

    def get_filter_codebook(self, weight_torch, compress_rate, length):
        ''' Gets filter codebook. '''
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0
          
            print("Filter codebook done.")
        else:
            pass
        return codebook

    def convert2tensor(self, x):
        ''' Converts an input to PyTorch tensor. '''
        x = torch.FloatTensor(x)
        return x

    def init_length(self):
        ''' Initializes the length of each layer. '''
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()

        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]

    def init_rate(self, layer_rate):
        ''' Initializes the compression rate of each layer. '''
        for index, item in enumerate(self.model.parameters()):
            self.compress_rate[index] = 1
        for key in range(self.args.layer_begin, self.args.layer_end + 1, self.args.layer_inter):
            self.compress_rate[key] = layer_rate
      
        # Last index includes last fully connected layer.
        last_index = 0
        skip_list = []
      
        if self.args.arch == 'resnet20':
            last_index = 57
        elif self.args.arch == 'resnet32':
            last_index = 93
        elif self.args.arch == 'resnet44':
            last_index = 129
        elif self.args.arch == 'resnet56':
            last_index = 165
        elif self.args.arch == 'resnet110':
            last_index = 327
        elif self.args.arch == 'resnet1202':
            last_index = 3603

        elif self.args.arch == 'resnet18':
            last_index = 60
            skip_list = [21, 36, 51]
        elif self.args.arch == 'resnet34':
            last_index = 108
            skip_list = [27, 54, 93]
        elif self.args.arch == 'resnet50':
            last_index = 159
            skip_list = [12, 42, 81, 138]
        elif self.args.arch == 'resnet101':
            last_index = 312
            skip_list = [12, 42, 81, 291]
        elif self.args.arch == 'resnet152':
            last_index = 465
            skip_list = [12, 42, 117, 444]
          
        self.mask_index = [x for x in range(0, last_index, 3)]

        # Skips downsample layer.
        if self.args.skip_downsample == 1:
            for x in skip_list:
                self.compress_rate[x] = 1
                self.mask_index.remove(x)

    def init_mask(self, layer_rate):
        ''' Initializes the mask. '''
        self.init_rate(layer_rate)
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index],
                                                           self.model_length[index])
                self.mat[index] = self.convert2tensor(self.mat[index])
                self.mat[index] = self.mat[index].to(self.device)
        print("Mask ready.")

    def do_mask(self):
        ''' Performs pruning. '''
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.model_size[index])
        print("Mask done.")

    def if_zero(self):
        ''' Prints information about network weights. '''
        for index, item in enumerate(self.model.parameters()):
            if index in [x for x in range(self.args.layer_begin, self.args.layer_end + 1, self.args.layer_inter)]:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()

                print("Layer: %d, number of nonzero weight is %d, zero is %d" % (
                      index, np.count_nonzero(b), len(b) - np.count_nonzero(b)))
