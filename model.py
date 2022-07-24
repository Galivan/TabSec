# Misc
import numpy as np
import pandas as pd
# Pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# Keras
import keras

class Net(nn.Module):
        def __init__(self, D_in, H, D_out, layers_size):

            """
            Neural Network
            :param D_in: input layer dimension
            :param H: hidden layer dimension
            :param D_out: output layer dimension
            :param layers_size: amount of hidden layers
            """

            super(Net, self).__init__()
            self.linears = nn.ModuleList([nn.Linear(D_in, H)])
            self.linears.extend(nn.ModuleList([nn.Linear(H, H) for i in range(1,layers_size - 1)]))
            self.linears.extend(nn.ModuleList([nn.Linear(H, D_out)]))
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=0)
            
        def forward(self, x):
            for linear in self.linears:
                x = self.relu(linear(x))
            y = self.softmax(x)
            return y