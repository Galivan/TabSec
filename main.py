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
                'n_train_adv': 50
                }
    test_string = "epochs={0}, lr={1}, scale_max={2}, alpha={3},\n" \
                  " lambda={4}, max_iters={5}".format(settings['epochs'], settings['lr'], settings['scale_max'],
                                                      settings['Alpha'], settings['Lambda'], settings['MaxIters'])
    settings['test_string'] = test_string
    torch.manual_seed(SEED)
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

    # Generate adversarial examples
    print("Generating adversarial examples for training...")
    orig_examples, df_adv_lpf, *lpf_train_data = gen_adv(nn_model, settings, 'LowProFool',
                                                         settings['TrainData'], n=settings['n_train_adv'])
    #df_adv_df, *df_train_data = gen_adv(nn_model, settings, 'Deepfool', settings['TrainAdv'])

    print("Generating adversarial examples for testing...")
    orig_examples, df_adv_lpf_test, *lpf_test_data = gen_adv(nn_model, settings, 'LowProFool',
                                                             settings['TestData'], n=settings['n_train_adv']*0.5)
    #df_adv_df_test, *df_test_data = gen_adv(nn_model, settings, 'Deepfool', settings['TestAdv'])

    real_sr, adv_sr = defense.test_svm_discriminator(settings, nn_model, 'LowProFool', settings['TestData'], orig_examples, df_adv_lpf_test)


    lpf_test_data.extend(test_data)
    #df_test_data.extend(test_data)
    #settings['AdvData'] = {'LowProFool': df_adv_lpf, 'Deepfool': df_adv_df}

    # Test fine-tuning method on LowProFool
    print("Testing fine tuning on LowProFool...")
    ft_lpf_model_clone = copy.deepcopy(nn_model)
    defense.test_fine_tune_low_pro_fool(ft_lpf_model_clone, device, test_dataloader, df_adv_lpf, lpf_test_data, settings)
    real_sr_ft, adv_sr_ft = defense.test_svm_discriminator(settings, ft_lpf_model_clone, 'LowProFool', settings['TrainData'], orig_examples, df_adv_lpf_test)

    print(f"Success rate before FT: real data - {real_sr}, adv data - {adv_sr}")
    print(f"Success rate after FT: real data - {real_sr_ft}, adv data - {adv_sr_ft}")
# Test fine-tuning method on Deep Fool
    print("Testing fine tuning on Deep Fool...")
    #ft_df_model_clone = copy.deepcopy(nn_model)
    #defense.test_fine_tune_deep_fool(ft_df_model_clone, device, test_dataloader, df_adv_df, df_test_data, settings)
    #defense.test_svm_discriminator(settings, ft_df_model_clone, 'Deepfool', settings['TrainAdv'], settings['TestAdv'], df_adv_df_test)




if __name__ == "__main__":
    main()
