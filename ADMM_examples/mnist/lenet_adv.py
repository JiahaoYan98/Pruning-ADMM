from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
from numpy import linalg as LA
import sys

class LeNet_adv(nn.Module):
    def __init__(self,w = 16,rho = 0.001):
        super(LeNet_adv, self).__init__()

        #p defined as partition size for channels, can be 2,4,6,8,16
        self.w = w
        print('LeNet width multiplier :',w)
        self.conv1 = nn.Conv2d(1, 2*self.w, 5, 1, 2)
        self.conv1_bn = nn.BatchNorm2d(2*self.w)
        self.conv2 = nn.Conv2d(2*self.w, 4*self.w, 5, 1, 2)
        self.conv2_bn = nn.BatchNorm2d(4*self.w)
        self.fc1 = nn.Linear(7*7*4*self.w, 64*self.w)
        self.fc1_bn = nn.BatchNorm1d(64*self.w)
        self.fc2 = nn.Linear(64*self.w, 10)

    def forward(self, x):
        #x = F.relu(self.conv1(x))
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 7*7*4*self.w)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
