import copy

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import metrics
import tester
import trainer
from adverse import gen_adv
import adverse_tabnet
from data_preparation import prepare_data
from def_finetune import test_fine_tuning
from def_tabnet_finetune import test_fine_tuning_tabnet
from def_svm_dicriminator import SVMDiscriminator, TabnetSVMDiscriminator
from def_svm_model import SvmModel, TabnetSvmModel
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

def test_tabnet_svm_discriminator(settings, model, adv_method, train_df, real_test_df, adv_test_df,
                           is_weighted=True, c=1.0, kernel='rbf', degree=3, gamma='scale'):
    svm_discriminator = TabnetSVMDiscriminator(settings, model, adv_method, is_weighted, c, kernel, degree, gamma)
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


def test_tabnet_svm_model(settings, model, adv_method, train_df, adv_df, benign_df,
                   is_weighted=True, c=1.0, kernel='rbf', degree=3, gamma='scale'):

    adv_df['target'] = benign_df['target'].values
    feature_names, target = settings['FeatureNames'], settings['Target']
    adv_test_loader = DataLoader(TabularDataset(adv_df, feature_names, target, False), batch_size=1)
    benign_test_loader = DataLoader(TabularDataset(benign_df, feature_names, target, False), batch_size=1)

    total_len = len(adv_test_loader) + len(benign_test_loader)
    svm_discriminator = TabnetSVMDiscriminator(settings, model, adv_method, is_weighted, c, kernel, degree, gamma)
    svm_discriminator.train(train_df)
    svm_model = TabnetSvmModel(settings, model, svm_discriminator)

    svm_model_adv_acc = svm_model.test(adv_df)
    svm_model_benign_acc = svm_model.test(benign_df)
    svm_model_total_acc = (svm_model_adv_acc * len(adv_test_loader) +
                           svm_model_benign_acc * len(benign_test_loader)) / total_len

    model_adv_acc = tester.test_tabnet_model(model, settings, adv_df)
    model_benign_acc = tester.test_tabnet_model(model, settings, benign_df)
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
                  " lambda={4}, max_iters={5}, method={6}".format(settings['epochs'],
                                                                  settings['lr'],
                                                                  settings['scale_max'],
                                                                  settings['Alpha'],
                                                                  settings['Lambda'],
                                                                  settings['MaxIters'],
                                                                  method)
    settings['test_string'] = test_string
    plt.figure(figsize=(15, 10))

    nn_model = Net(dimensions[0], settings['hidden_dim'], dimensions[1], settings['layers'])
    settings["Model"] = nn_model

    train_losses, train_accuracies = train_bce_adam_model(nn_model, device, train_dataloader, settings['lr'], settings['epochs'])
    test_data = tester.test_bce_model(nn_model, device, test_dataloader)
    print(test_data)

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
    orig_examples_lpf_test, df_adv_lpf_test, *lpf_test_data = gen_adv(nn_model, settings, method,
                                                                      settings['TestData'], n=settings['n_train_adv'])
    is_weighted = True if method == 'LowProFool' else False
    print("Testing SVM Accuracy")
    real_sr, adv_sr = test_svm_discriminator(settings, nn_model, method,
                                             settings['TrainData'], orig_examples_lpf_test, df_adv_lpf_test, is_weighted)

    print(f"Real SR = {real_sr}, adv SR = {adv_sr}")
    test_svm_model(settings, device, nn_model, method, settings['TrainData'],
                           df_adv_lpf_test, orig_examples_lpf_test)

    lpf_test_data.extend(test_data)
    settings['AdvData'] = {method: df_adv_lpf}

    print(f"Testing fine tuning on {method}...")
    ft_lpf_model_clone = copy.deepcopy(nn_model)
    test_fine_tuning(ft_lpf_model_clone, device, test_dataloader, df_adv_lpf_test, lpf_test_data,method, settings)
    real_sr_ft, adv_sr_ft = test_svm_discriminator(settings, ft_lpf_model_clone, method, settings['TrainData'], orig_examples_lpf_test, df_adv_lpf_test)

    print(f"Success rate after FT: real data - {real_sr_ft}, adv data - {adv_sr_ft}")

def def_tabnet_model(settings, device, method):
    test_string = "epochs={0}, lr={1}, scale_max={2}, alpha={3},\n" \
                  " lambda={4}, max_iters={5}, method={6}".format(settings['epochs'],
                                                                  settings['lr'],
                                                                  settings['scale_max'],
                                                                  settings['Alpha'],
                                                                  settings['Lambda'],
                                                                  settings['MaxIters'],
                                                                  method)
    settings['test_string'] = test_string
    settings['dataset_name'] = 'credit-g'
    settings['Target'] = 'target'
    plt.figure(figsize=(15, 10))

    settings = prepare_data(settings)
    tabnet_model = trainer.get_trained_tabnet_model(settings)
    settings["Model"] = tabnet_model

    tabnet_acc = tester.test_tabnet_model(tabnet_model, settings, settings['TestData'])



    # Generate adversarial examples
    print("Generating adversarial examples for training...")
    orig_examples_train, df_adv_train, *lpf_train_data = adverse_tabnet.gen_adv(tabnet_model, settings, method,
                                                                                settings['TrainData'],
                                                                                n=settings['n_train_adv'])

    print("Generating adversarial examples for testing...")
    orig_examples_test, df_adv_test, *lpf_test_data = adverse_tabnet.gen_adv(tabnet_model, settings, method,
                                                                             settings['TestData'],
                                                                             n=settings['n_train_adv'])
    is_weighted = True if method == 'LowProFool' else False
    lpf_test_data.append(tabnet_acc)
    print("Testing SVM Accuracy")
    real_sr, adv_sr = test_tabnet_svm_discriminator(settings, tabnet_model, method,
                                                    settings['TrainData'], orig_examples_test,
                                                    df_adv_test, is_weighted)

    print(f"Real SR = {real_sr}, adv SR = {adv_sr}")
    test_tabnet_svm_model(settings, tabnet_model, method, settings['TrainData'],
                   df_adv_test, orig_examples_test)

    settings['AdvData'] = {method: df_adv_train}

    print(f"Testing fine tuning on {method}...")
    ft_model_clone = copy.deepcopy(tabnet_model)
    test_fine_tuning_tabnet(ft_model_clone, device, settings['TestData'], df_adv_test, lpf_test_data, method, settings)
    test_tabnet_svm_model(settings, ft_model_clone, method, settings['TrainData'],
                          df_adv_test, orig_examples_test)



