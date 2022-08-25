from torch.utils.data import DataLoader

import metrics
from def_finetune import test_fine_tuning
from def_svm_dicriminator import SVMDiscriminator
from def_svm_model import SvmModel
from tabular_dataset import TabularDataset
from tester import test_bce_model


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
    metrics.show_table("SVM Boosted Model Results",
                       ["Adv. Acc.", "Benign Acc.", "Total Acc."],
                       ["Original Model", "SVM Boosted Model"],
                       data)




