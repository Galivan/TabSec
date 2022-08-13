import copy

import numpy as np
import torch.nn
import matplotlib.pyplot as plt

import dataset_factory

from torch.utils.data import DataLoader

import defense
import metrics
import tester
from model import Net
from tabular_dataset import TabularDataset
from trainer import Trainer, train_bce_adam_model
from adverse import gen_adv


from IPython.display import display

SEED = 0


def main():
    # + configurations for adversarial generation
    settings = {'batch_size': 100,
                'epochs': 500,
                'hidden_dim': 100,
                'layers': 5,
                'lr': 1e-4,
                'MaxIters': 20000,
                'Alpha': 0.001,
                'Lambda': 8.5
                }

    plt.figure(figsize=(15, 10))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader, test_dataloader, dimensions = dataset_factory.get_credit_g_dataloaders(settings)
    nn_model = Net(dimensions[0], settings['hidden_dim'], dimensions[1], settings['layers'])
    settings["Model"] = nn_model

    train_losses, train_accuracies = train_bce_adam_model(nn_model, device, train_dataloader, settings['lr'], settings['epochs'])
    test_data = tester.test_bce_model(nn_model, device, test_dataloader)

    fig, axs = plt.subplots(2, 1)
    x = np.arange(settings['epochs'])
    axs[0].plot(x, train_losses, label="train_losses")
    axs[0].set_title("train_losses")
    axs[1].plot(x, train_accuracies, label="train_accuracies")
    axs[1].set_title("train_accuracies")
    plt.show()

    # Sub sample
    settings['TestData'] = settings['TestData'].sample(n=2, random_state=SEED)

    # Generate adversarial examples
    df_adv_lpf, *lpf_data = gen_adv(nn_model, settings, 'LowProFool')
    # display(df_adv_lpf)
    df_adv_df, *df_data = gen_adv(nn_model,settings, 'Deepfool')
    lpf_data.extend(test_data)
    df_data.extend(test_data)
    settings['AdvData'] = {'LowProFool': df_adv_lpf, 'Deepfool': df_adv_df}

    # Test fine-tuning method on LowProFool
    ft_lpf_model_clone = copy.deepcopy(nn_model)
    defense.test_fine_tune_low_pro_fool(ft_lpf_model_clone, device, test_dataloader, df_adv_lpf, lpf_data, settings)

    # Test fine-tuning method on DeepFool
    ft_df_model_clone = copy.deepcopy(nn_model)
    defense.test_fine_tune_deep_fool(ft_df_model_clone, device, test_dataloader, df_adv_df, df_data, settings)


if __name__ == "__main__":
    main()
