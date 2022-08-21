
import copy
import itertools
import numpy as np
import torch.nn
import matplotlib.pyplot as plt
import os
import pandas as pd

import dataset_factory

import tester
from model import Net
from trainer import train_bce_adam_model
from adverse import gen_adv
from def_finetune import test_fine_tuning
import defense

SEED = 0


def cross_test():
    columns = ["real sr", "adv sr", "epochs", "learning rate", "scale_max", "alpha", "lambda", "max_iters"]
    svm_results = pd.DataFrame(columns=columns)
    try:
        try:
            os.mkdir("LowProFool")
        except OSError:
            pass
        try:
            os.mkdir("LowProFool/svm")
        except OSError:
            pass
        epochs = [300, 500]
        lrs = [0.0001, 0.001]
        scale_max = [50]
        alphas = [0.001, 0.005, 0.01]
        lambdas = [8.5]
        max_iters = [200, 800, 2000]
        n_trains = [50]
        n = 0
        for (epoch, lr, scale, alpha, lambd, max_iter, n_train) in itertools.product(epochs, lrs, scale_max, alphas,
                                                                                     lambdas, max_iters, n_trains):
            print(f"Test number {n}")
            #cross_test_fine_tuning('LowProFool', epoch, lr, scale, alpha, lambd, max_iter, n_train, f"LowProFool/{n}")
            real_sr, adv_sr = cross_test_svm_disc('LowProFool', epoch, lr, scale, alpha, lambd, max_iter, n_train, f"LowProFool/svm/{n}")
            svm_results.loc[n] = [real_sr, adv_sr, epoch, lr, scale, alpha, lambd, max_iter]
            n += 1
    finally:
        svm_results.to_csv("LowProFool/svm/results.csv")
        print(svm_results)
def cross_test_fine_tuning(method, epochs, lr, scale_max, alpha, lambd, max_iters, n_train, file_name):
    # + configurations for adversarial generation
    test_string = "epochs={0}, lr={1}, scale_max={2}, alpha={3},\n" \
                  " lambda={4}, max_iters={5}, n_train={6}".format(epochs, lr, scale_max, alpha, lambd, max_iters, n_train)
    print(test_string)
    settings = {'batch_size': 100,
                'epochs': epochs,
                'hidden_dim': 100,
                'layers': 5,
                'lr': lr,
                'MaxIters': max_iters,
                'Alpha': alpha,
                'Lambda': lambd,
                'scale_max': scale_max,
                'test_string': test_string
                }
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
    axs[0].set_title(test_string)
    axs[1].plot(x, train_accuracies, label="train_accuracies")
    plt.savefig(file_name+"_acc")

    # Sub sample
    settings['TrainAdv'] = settings['TrainData'].sample(n=n_train, random_state=SEED)
    settings['TestAdv'] = settings['TestData'].sample(n=int(0.3*n_train), random_state=SEED)

    # Generate adversarial examples
    #print("Generating adversarial examples for training...")
    adv_examples, *adv_train_data = gen_adv(nn_model, settings, method, settings['TrainAdv'])

    #print("Generating adversarial examples for testing...")
    adv_examples_test, *adv_test_data = gen_adv(nn_model, settings, method, settings['TestAdv'])
    adv_test_data.extend(test_data)

    # Test fine-tuning method on LowProFool
    #print("Testing fine tuning on LowProFool...")
    ft_model_clone = copy.deepcopy(nn_model)
    test_fine_tuning(ft_model_clone, device, test_dataloader, adv_examples, adv_test_data, method, settings)
    plt.savefig(file_name+"_table")


def cross_test_fine_tuning_deepfool():
    pass


def cross_test_svm_disc(method, epochs, lr, scale_max, alpha, lambd, max_iters, n_train, file_name):
    # + configurations for adversarial generation
    test_string = "epochs={0}, lr={1}, scale_max={2}, alpha={3},\n" \
                  " lambda={4}, max_iters={5}".format(epochs, lr, scale_max, alpha, lambd, max_iters, )
    print(test_string)
    settings = {'batch_size': 100,
                'epochs': epochs,
                'hidden_dim': 100,
                'layers': 5,
                'lr': lr,
                'MaxIters': max_iters,
                'Alpha': alpha,
                'Lambda': lambd,
                'scale_max': scale_max,
                'test_string': test_string
                }
    torch.manual_seed(SEED)
    plt.figure(figsize=(15, 10))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader, test_dataloader, dimensions = dataset_factory.get_credit_g_dataloaders(settings)
    nn_model = Net(dimensions[0], settings['hidden_dim'], dimensions[1], settings['layers'])
    settings["Model"] = nn_model

    train_losses, train_accuracies = train_bce_adam_model(nn_model, device, train_dataloader, settings['lr'], settings['epochs'])
    test_data = tester.test_bce_model(nn_model, device, test_dataloader)

    # Sub sample
    settings['TrainAdv'] = settings['TrainData'].sample(n=n_train, random_state=SEED)
    settings['TestAdv'] = settings['TestData'].sample(n=int(0.5*n_train), random_state=SEED)

    adv_examples_test, *adv_test_data = gen_adv(nn_model, settings, method, settings['TestAdv'])
    adv_test_data.extend(test_data)

    return defense.test_svm_discriminator(settings, nn_model, method, settings['TrainAdv'], settings['TestAdv'], adv_examples_test)



if __name__ == "__main__":
    cross_test()
