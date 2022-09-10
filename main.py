import copy

import numpy as np
import torch.nn
import matplotlib.pyplot as plt

import dataset_factory

from torch.utils.data import DataLoader

import defense
import metrics
import tester
from def_bouncer import test_bouncer
from model import Net
from tabular_dataset import TabularDataset
from trainer import Trainer, train_bce_adam_model
from adverse import gen_adv


from IPython.display import display

SEED = 0
lowProFool = 'LowProFool'
deepfool = 'Deepfool'

def main():
    # + configurations for adversarial generation
    normal_settings = {'batch_size': 100,
                       'epochs': 100,
                       'hidden_dim': 100,
                       'layers': 5,
                       'lr': 0.001,
                       'MaxIters': 2000,
                       'Alpha': 0.001,
                       'Lambda': 8.5,
                       'scale_max': 10,
                       'n_train_adv': 100
                       }
    tabnet_settings = {'batch_size': 100,
                       'epochs': 500,
                       'hidden_dim': 100,
                       'layers': 5,
                       'lr': 0.1,
                       'MaxIters': 100,
                       'Alpha': 0.001,
                       'Lambda': 8.5,
                       'scale_max': 10,
                       'n_train_adv': 15,
                       'Seed': SEED
                       }
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    plt.figure(figsize=(15, 10))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader, test_dataloader, dimensions = dataset_factory.get_credit_g_dataloaders(normal_settings)

    #defense.test_normal_model(normal_settings, device, train_dataloader, test_dataloader, dimensions, lowProFool)
    #defense.test_normal_model(normal_settings, device, train_dataloader, test_dataloader, dimensions, deepfool)
    defense.def_tabnet_model(tabnet_settings, device, lowProFool)
    defense.def_tabnet_model(tabnet_settings, device, deepfool)




if __name__ == "__main__":
    main()
