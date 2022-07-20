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

def get_model(conf, layers_size=3,  load=False):
    assert(conf['Dataset'] == 'credit-g')
    assert(layers_size >= 2)

    class Net(nn.Module):
        def __init__(self, D_in, H, D_out, layers_size):
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
        
    def train(model, criterion, optimizer, X, y, N, n_classes):
        pass
    
    df = conf['TrainData']
    target = conf['Target']
    feature_names = conf['FeatureNames']
    
    n_classes = len(np.unique(df[target]))
    X_train = torch.FloatTensor(df[feature_names].values)
    y_train = keras.utils.to_categorical(df[target], n_classes)
    y_train = torch.FloatTensor(y_train)
    
    D_in = X_train.size(1)
    D_out = y_train.size(1)
    
    epochs = 400
    batch_size = 100
    H = 100
    layers_size = 5
    net = Net(D_in, H, D_out, layers_size)
    # net = GermanNet(D_in, H, D_out)
    
    lr = le-4
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    for epoch in range(epochs):
        preds, epoch_loss, epoch_acc = train(net, criterion, optimizer, X_train, y_train, batch_size, n_classes)
        if (epoch % 50 == 0):
            print("> epoch {:.0f}\tLoss {:.5f}\tAcc {:.5f}".format(epoch, epoch_loss, epoch_acc))
            
    net.eval()
    
    return net