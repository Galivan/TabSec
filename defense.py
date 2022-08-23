from def_finetune import test_fine_tuning
from def_svm_dicriminator import SVMDiscriminator


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




