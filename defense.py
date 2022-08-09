import torch
from torch.utils.data import DataLoader

import metrics
from adverse import gen_adv
from tabular_dataset import TabularDataset
from trainer import train_bce_adam_model
from tester import test_bce_model

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


def test_fine_tuning(model, device, test_dataloader, adv_examples, adv_data_before, adv_name, settings):
    fine_tune_model(model, device, adv_examples, settings)
    test_data_after_ft = test_bce_model(model, device, test_dataloader)
    df_adv_lpf, *lpf_data_after_ft = gen_adv(model, settings, adv_name)
    lpf_data_after_ft.extend(test_data_after_ft)
    metrics.plot_metrics(adv_data_before, lpf_data_after_ft, f"{adv_name} - Fine Tuning")

def test_fine_tune_low_pro_fool(nn_model, device, test_dataloader, adv_examples, adv_data_before, settings):
    test_fine_tuning(nn_model, device, test_dataloader, adv_examples, adv_data_before, "LowProFool", settings)

def test_fine_tune_deep_fool(nn_model, device, test_dataloader, adv_examples, adv_data_before, settings):
    test_fine_tuning(nn_model, device, test_dataloader, adv_examples, adv_data_before, "Deepfool", settings)