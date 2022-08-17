import torch
from torch.utils.data import DataLoader


import metrics
from adverse import gen_adv
from tabular_dataset import TabularDataset
from trainer import train_bce_adam_model
from tester import test_bce_model
from def_finetune import test_fine_tuning

ft_settings = {'lr_ratio': 0.1,
               'epochs_ratio': 1}




def test_fine_tune_low_pro_fool(nn_model, device, test_dataloader, adv_examples, adv_data_before, settings):
    """
    Test fine tune method of LowProFool
    :param nn_model: NN Model for the data
    :param device: pytorch device
    :param test_dataloader: Test data pytorch Dataloader
    :param adv_examples: Adversarial examples created by LPF
    :param adv_data_before: Measurements data before fine tuning
    :param settings: General settings
    :return: None
    """
    test_fine_tuning(nn_model, device, test_dataloader, adv_examples, adv_data_before, "LowProFool", settings)

def test_fine_tune_deep_fool(nn_model, device, test_dataloader, adv_examples, adv_data_before, settings):
    """
    Test fine tune method of DeepFool
    :param nn_model: NN Model for the data
    :param device: pytorch device
    :param test_dataloader: Test data pytorch Dataloader
    :param adv_examples: Adversarial examples created by DF
    :param adv_data_before: Measurements data before fine tuning
    :param settings: General settings
    :return: None
    """
    test_fine_tuning(nn_model, device, test_dataloader, adv_examples, adv_data_before, "Deepfool", settings)