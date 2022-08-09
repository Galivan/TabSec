import torch
from torch.utils.data import DataLoader

from tabular_dataset import TabularDataset
from trainer import train_bce_adam_model

ft_settings = {'lr_ratio': 0.1,
               'epochs_ratio': 1}


def fine_tune_model(model, device, adverse_data, settings):
    """
    Fine-tune a model using adverse examples as training data.
    :param model: Model to fine-tune
    :param device: torch device(Cuda or CPU)
    :param adverse_data: Dataframe of adversarial data. Target should be the desired "True" target.
    :param settings: Settings of the main model - used for epochs and lr

    :return: training loss and training accuracies of the fine-tuning process
    """
    ft_dataset = TabularDataset(adverse_data, settings['FeatureNames'], settings['Target'], True)
    ft_dataloader = DataLoader(ft_dataset, batch_size=1, shuffle=True)

    ft_lr = ft_settings['lr_ratio'] * settings['lr']
    ft_epochs = ft_settings['epochs_ratio'] * settings['epochs']

    return train_bce_adam_model(model, device, ft_dataloader, ft_lr, ft_epochs)