import numpy as np
import torch
import keras
from torch.utils.data import DataLoader

import metrics
import pandas as pd
from model import Net
from adverse import gen_adv
from tabular_dataset import TabularDataset
from trainer import train_bce_adam_model
from tester import test_bce_model


bouncer_settings = {'lr': 1e-4,
                    'epochs': 500}


def get_trained_bouncer_model(device, true_data_df, adv_data_df, settings):
    features = np.append(settings['FeatureNames'], (settings['Target']))
    train_dataloader, dimensions = get_bouncer_dataloaders(true_data_df, adv_data_df, features)
    model = Net(dimensions[0], settings['hidden_dim'], dimensions[1], settings['layers'])
    train_bce_adam_model(model, device, train_dataloader, bouncer_settings['lr'], bouncer_settings['epochs'])
    return model


def get_bouncer_dataloaders(true_data_df, adv_data_df, features):
    target = "is_adv"
    true_data_df = true_data_df.assign(is_adv=0)
    adv_data_df = adv_data_df.assign(is_adv=1)
    training_df = pd.concat([true_data_df, adv_data_df])
    mixed_dataset = TabularDataset(training_df, features, target, True)
    mixed_dataloader = DataLoader(mixed_dataset, batch_size=100, shuffle=True)
    return mixed_dataloader, mixed_dataset.get_dimensions()


def test_bouncer(device, true_data_test_df, adv_data_test_df, adv_examples, settings):
    bouncer = get_trained_bouncer_model(device, settings['TrainData'], adv_examples, settings)
    target = "is_adv"
    features = np.append(settings['FeatureNames'], (settings['Target']))

    true_data_test_df = true_data_test_df.assign(is_adv=0)
    adv_data_test_df = adv_data_test_df.assign(is_adv=1)

    true_test_dataset = TabularDataset(true_data_test_df, features, target, False)
    adv_test_dataset = TabularDataset(adv_data_test_df, features, target, False)

    true_test_dataloader = DataLoader(true_test_dataset, batch_size=len(true_test_dataset))
    adv_test_dataloader = DataLoader(adv_test_dataset, batch_size=len(adv_test_dataset))

    bouncer_test_real_data = test_bce_model(bouncer, device, true_test_dataloader)
    bouncer_test_adv_data = test_bce_model(bouncer, device, adv_test_dataloader)

    print(f"Bouncer on real data: {bouncer_test_real_data}")
    print(f"Bouncer on adversarial data: {bouncer_test_adv_data}")

