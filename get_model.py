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
# Neural Network
from model import Net

def get_model(conf, layers_size=3,  load=False):

    """
    Generates a Neural Network and trains it on dataset

    :param conf: preproccessed dataset configuration
    :layers_size: amount of hidden layers
    :return: a trained neural network
    """
    assert(conf['Dataset'] == 'credit-g')
    assert(layers_size >= 2)

    def train(model, criterion, optimizer, X, y, N, n_classes):
        model.train()

        current_loss = 0
        current_correct = 0


        # Training in batches
        for ind in range(0, X.size(0), N):
            indices = range(ind, min(ind + N, X.size(0)) - 1) 
            inputs, labels = X[indices], y[indices]
            inputs = Variable(inputs, requires_grad=True)


            optimizer.zero_grad()

            output = model(inputs)
            _, indices = torch.max(output, 1) # argmax of output [[0.61, 0.12]] -> [0]
            # [[0, 1, 1, 0, 1, 0, 0]] -> [[1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0], [1, 0]]
            preds = torch.tensor(keras.utils.to_categorical(indices, num_classes=n_classes))

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()
            current_correct += (preds.int() == labels.int()).sum() /n_classes


        current_loss = current_loss / X.size(0)
        current_correct = current_correct.double() / X.size(0)    

        return preds, current_loss, current_correct.item()
    
    df = conf['TrainData']
    target = conf['Target']
    feature_names = conf['FeatureNames']
    
    n_classes = len(np.unique(df[target]))
    X_train = torch.DoubleTensor(df[feature_names].values)
    y_train = keras.utils.to_categorical(df[target], n_classes)
    y_train = torch.DoubleTensor(y_train)
    
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