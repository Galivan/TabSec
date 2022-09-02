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


def main():
    # + configurations for adversarial generation
    settings = {'batch_size': 100,
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
    test_string = "epochs={0}, lr={1}, scale_max={2}, alpha={3},\n" \
                  " lambda={4}, max_iters={5}".format(settings['epochs'], settings['lr'], settings['scale_max'],
                                                      settings['Alpha'], settings['Lambda'], settings['MaxIters'])
    settings['test_string'] = test_string
    torch.manual_seed(SEED)
    plt.figure(figsize=(15, 10))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader, test_dataloader, dimensions = dataset_factory.get_credit_g_dataloaders(settings)

    defense.test_normal_model(settings, device, train_dataloader, test_dataloader, dimensions, 'LowProFool')


if __name__ == "__main__":
    main()
