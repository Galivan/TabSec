import copy

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import metrics
import tester
from adverse import gen_adv
from def_finetune import test_fine_tuning
from def_svm_dicriminator import SVMDiscriminator
from def_svm_model import SvmModel
from model import Net
from tabular_dataset import TabularDataset
from tester import test_bce_model
from trainer import train_bce_adam_model


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


def test_svm_discriminator(settings, model, adv_method, train_df, real_test_df, adv_test_df,
                           is_weighted=True, c=1.0, kernel='rbf', degree=3, gamma='scale'):
    svm_discriminator = SVMDiscriminator(settings, model, adv_method, is_weighted, c, kernel, degree, gamma)
    svm_discriminator.train(train_df)
    return svm_discriminator.test(real_test_df, adv_test_df)


def test_svm_model(settings, device, model, adv_method, train_df, adv_df, benign_df,
                   is_weighted=True, c=1.0, kernel='rbf', degree=3, gamma='scale'):

    adv_df['target'] = benign_df['target'].values
    feature_names, target = settings['FeatureNames'], settings['Target']
    adv_test_loader = DataLoader(TabularDataset(adv_df, feature_names, target, False), batch_size=1)
    benign_test_loader = DataLoader(TabularDataset(benign_df, feature_names, target, False), batch_size=1)

    total_len = len(adv_test_loader) + len(benign_test_loader)
    svm_discriminator = SVMDiscriminator(settings, model, adv_method, is_weighted, c, kernel, degree, gamma)
    svm_discriminator.train(train_df)
    svm_model = SvmModel(settings, model, svm_discriminator)

    svm_model_adv_acc = svm_model.test(adv_df)
    svm_model_benign_acc = svm_model.test(benign_df)
    svm_model_total_acc = (svm_model_adv_acc * len(adv_test_loader) +
                           svm_model_benign_acc * len(benign_test_loader)) / total_len

    _, model_adv_acc = test_bce_model(model, device, adv_test_loader)
    _, model_benign_acc = test_bce_model(model, device, benign_test_loader)
    model_total_acc = (model_adv_acc * len(adv_test_loader) +
                       model_benign_acc * len(benign_test_loader)) / total_len
    data = [[model_adv_acc, model_benign_acc, model_total_acc],
            [svm_model_adv_acc, svm_model_benign_acc, svm_model_total_acc]]
    adv_p = (len(adv_test_loader)/total_len)*100
    metrics.show_table(f"SVM Boosted Model Results - {adv_p:.2f}% adversarial ",
                       ["Adv. Acc.", "Benign Acc.", "Total Acc."],
                       ["Original Model", "SVM Boosted Model"],
                       data)


def test_normal_model(settings, device, train_dataloader, test_dataloader, dimensions, method):
    # + configurations for adversarial generation
    test_string = "epochs={0}, lr={1}, scale_max={2}, alpha={3},\n" \
                  " lambda={4}, max_iters={5}".format(settings['epochs'], settings['lr'], settings['scale_max'],
                                                      settings['Alpha'], settings['Lambda'], settings['MaxIters'])
    settings['test_string'] = test_string
    plt.figure(figsize=(15, 10))

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
    orig_examples, df_adv_lpf, *lpf_train_data = gen_adv(nn_model, settings, method,
                                                          settings['TrainData'], n=settings['n_train_adv'])

    print("Generating adversarial examples for testing...")
    orig_examples_lpf_test, df_adv_lpf_test, *lpf_test_data = gen_adv(nn_model, settings, 'LowProFool',
                                                                      settings['TestData'], n=settings['n_train_adv'])

    real_sr, adv_sr = test_svm_discriminator(settings, nn_model, method,
                                             settings['TrainData'], orig_examples, df_adv_lpf_test)
    test_svm_model(settings, device, nn_model, "LowProFool", settings['TrainData'],
                           df_adv_lpf_test, orig_examples_lpf_test)

    lpf_test_data.extend(test_data)
    settings['AdvData'] = {method: df_adv_lpf}

    # Test fine-tuning method on LowProFool
    print("Testing fine tuning on LowProFool...")
    ft_lpf_model_clone = copy.deepcopy(nn_model)
    test_fine_tuning(ft_lpf_model_clone, device, test_dataloader, df_adv_lpf_test, lpf_test_data,method, settings)
    real_sr_ft, adv_sr_ft = test_svm_discriminator(settings, ft_lpf_model_clone, 'LowProFool', settings['TrainData'], orig_examples_lpf_test, df_adv_lpf_test)

    print(f"Success rate after FT: real data - {real_sr_ft}, adv data - {adv_sr_ft}")


