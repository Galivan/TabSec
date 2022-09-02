import numpy as np
from torch.utils.data import DataLoader

import metrics
from adverse import gen_adv
from tabular_dataset import TabularDataset
from trainer import train_bce_adam_model
from tester import test_bce_model


ft_settings = {'lr_ratio': 0.1,
               'epochs_ratio': 1,
               'batch_size': 100}


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
    ft_dataloader = DataLoader(ft_dataset, batch_size=ft_settings['batch_size'], shuffle=True)

    ft_lr = ft_settings['lr_ratio'] * settings['lr']
    ft_epochs = int(ft_settings['epochs_ratio'] * settings['epochs'])

    return train_bce_adam_model(model, device, ft_dataloader, ft_lr, ft_epochs)


def test_fine_tuning(model, device, test_dataloader, adv_examples, adv_data_before, adv_name, settings):
    """
    Test fine tuning method on adversarial data
    :param nn_model: NN Model for the data
    :param device: pytorch device
    :param test_dataloader: Test data pytorch Dataloader
    :param adv_examples: Adversarial examples created by adversarial algorithm
    :param adv_data_before: Measurements data before fine tuning
    :param adv_name: Name of adversarial method
    :param settings: General settings
    :return: None
    """
    fine_tune_model(model, device, adv_examples, settings)
    test_data_after_ft = test_bce_model(model, device, test_dataloader)

    orig_examples, df_adv_lpf, *lpf_data_after_ft = gen_adv(model, settings, adv_name,
                                                            settings['TestData'], n=0.5*settings['n_train_adv'])
    lpf_data_after_ft.extend(test_data_after_ft)
    mean_data_before = np.vectorize(np.mean)(np.array(adv_data_before, dtype=object))
    mean_data_after = np.vectorize(np.mean)(np.array(lpf_data_after_ft, dtype=object))
    metrics.show_finetuning_metrics(adv_data_before, lpf_data_after_ft, settings['test_string'])
    return mean_data_after - mean_data_before
