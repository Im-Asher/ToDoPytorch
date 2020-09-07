import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutil

import matplotlib.pyplot as plt
import numpy as np

image_size = 28
input_dim = 100
num_channels =1
num_features = 64
batch_size = 64



class ModelD(nn.Module):
    def __init__(self):
        super(ModelD,self).__init__()
        self.model = nn.Sequential()
        self.model.add_module('conv1', nn.Conv2d(num_channels, num_features, 5, 2, 0, bias=False))
        self.model.add_module('relu1', nn.ReLU())

        # 第二层卷积
        self.model.add_module('conv2', nn.Conv2d(num_features, num_features * 2, 5, 2, 0, bias=False))
        self.model.add_module('bnorm2', nn.BatchNorm2d(num_features * 2))
        self.model.add_module('linear1', nn.Linear(num_features * 2 * 4 * 4, num_features))
        self.model.add_module('sigmoid', nn.Sigmoid())
    def forward(self, input):
        output = input
        for name, module in self.model.named_children():
            if name == 'linear1':
                output = output.view(-1, num_features * 2 * 4 *4)
            output = module(output)
        return output



