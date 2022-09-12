from pathlib import Path

import keras
import torch
import numpy as np
import os

from keras.utils import np_utils
from sklearn.metrics import roc_auc_score

from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.augmentations import ClassificationSMOTE


class Trainer:
    """
    Class used to train a FCNN model
    """
    def __init__(self, model, device, loss_function, optimizer):
        """
        :param model: Model to train. Needs to inherit torch.nn.Module
        :param device: torch.device. Use 'cuda' for faster computations if possible
        :param loss_function: Loss function to train with (best is to functions from torch.nn)
        :param optimizer: Optimizer to train with. Use torch.optim for it
        """
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model = model
        self.device = device
        self.train_losses = []
        self.train_accuracies = []

    def train(self, num_epochs, train_loader):
        """
        Trains the model, with regular backpropagation

        :param num_epochs: Number of epochs to run
        :param train_loader: Dataloader of type torch.utils.data.Dataloader for training.

        :return: None
        """
        for epoch in range(num_epochs):
            #if (epoch + 1) % 100 == 0:
                #print(f'Epoch [{epoch + 1}/{num_epochs}]...')
            n_correct = 0
            total_loss = 0
            for i, (data, labels) in enumerate(train_loader):
                # Forward pass
                labels = labels.to(self.device)
                predictions, loss_val = self.forward(data, labels)

                # Backward pass
                self.backward(loss_val)

                _, idxs = torch.max(predictions, 1)
                prediction_label = torch.tensor(keras.utils.np_utils.to_categorical(idxs.cpu(), num_classes=len(labels[0]))).to(self.device)
                n_correct += 0.5*(prediction_label == labels).sum().item()
                total_loss += loss_val.item()

            train_acc = n_correct / (len(train_loader) * train_loader.batch_size)
            train_loss = total_loss/len(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

    def forward(self, data, labels):
        """
        Forward pass of the data.
        :param data: Inputs to the model
        :param labels: True labels
        :return: (model predictions, loss on the data)
        """
        #data = data.to(self.device)
        predictions = self.model(data).to(self.device)
        loss_val = self.loss_function(predictions, labels)
        return predictions, loss_val

    def backward(self, loss_value):
        """
        Backwards pass for the backprop
        :param loss_value: value of the loss
        :return: None
        """
        loss_value.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_data(self):
        """
        Returns data of training
        :return: train_losses, train_acc
        """
        return self.train_losses, self.train_accuracies


def train_bce_adam_model(model, device, train_dataloader, lr, epochs):
    loss_func = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = Trainer(model, device, loss_func, optimizer)
    trainer.train(epochs, train_dataloader)
    return trainer.get_data()


def get_trained_tabnet_model(config):
    saving_path_name = "./tabnet_model_test_1_norm"
    out = Path(saving_path_name+".zip")
    out.parent.mkdir(parents=True, exist_ok=True)

    cat_idxs = config['cat_idxs']
    cat_dims = config['cat_dims']

    train_data = config['TrainData']
    validation_data = config['ValidData']
    features = config['FeatureNames']
    target = config['Target']

    tabnet_params = {"cat_idxs":cat_idxs,
                     "cat_dims":cat_dims,
                     "cat_emb_dim":1,
                     "optimizer_fn":torch.optim.Adam,
                     "optimizer_params":dict(lr=2e-3),
                     "scheduler_params":{"step_size":20, # how to use learning rate scheduler
                                         "gamma":0.9},
                     "scheduler_fn":torch.optim.lr_scheduler.StepLR,
                     "mask_type":'entmax' # "sparsemax"
                     }

    clf = TabNetClassifier(**tabnet_params
                           )
    if out.exists():
        print('Found saved model... loading')
        clf.load_model(saving_path_name+".zip")
        return clf

    X_train = train_data[features].values
    y_train = train_data[target].values

    X_valid = validation_data[features].values
    y_valid = validation_data[target].values

    max_epochs = config['epochs'] if not os.getenv("CI", False) else 2

    aug = ClassificationSMOTE(p=0.2)

    save_history = []

    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=['train', 'valid'],
        eval_metric=['auc'],
        max_epochs=max_epochs, patience=int(max_epochs/5),
        batch_size=1024, virtual_batch_size=128,
        num_workers=0,
        weights=1,
        drop_last=False,
        augmentations=aug, #aug, None
    )
    clf.save_model(saving_path_name)
    save_history.append(clf.history["valid_auc"])
    return clf
