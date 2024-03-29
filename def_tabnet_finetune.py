import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import metrics
from adverse_tabnet import gen_adv
from pytorch_tabnet.augmentations import ClassificationSMOTE
from pytorch_tabnet.tab_model import TabNetClassifier
from tabular_dataset import TabularDataset
from trainer import train_bce_adam_model
from tester import test_bce_model, test_tabnet_model

ft_settings = {'lr_ratio': 0.1,
               'epochs': 1,
               'batch_size': 100}


def fine_tune_tabnet(model, device, adverse_data, settings):
    """
    Fine-tune a model using adverse examples as training data.
    :param model: Model to fine-tune
    :param device: torch device(Cuda or CPU)
    :param adverse_data: Dataframe of adversarial data. Target should be the desired "True" target.
    :param settings: Settings of the main model - used for epochs and lr

    :return: training loss and training accuracies of the fine-tuning process
    """

    features = settings['FeatureNames']
    target = settings['Target']

    X = adverse_data[features]
    y = adverse_data[target]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=settings['Seed'], stratify=y)
    max_epochs = settings['epochs'] if not os.getenv("CI", False) else 2
    max_epochs=60
    aug = ClassificationSMOTE(p=0.2)
    cat_idxs = settings['cat_idxs']
    cat_dims = settings['cat_dims']
    tabnet_params = {"cat_idxs":cat_idxs,
                     "cat_dims":cat_dims,
                     "cat_emb_dim":1,
                     "optimizer_fn":torch.optim.Adam,
                     "optimizer_params":dict(lr=2e-3),
                     "scheduler_params":{"step_size":20, # how to use learning rate scheduler
                                         "gamma":0.9},
                     "scheduler_fn":torch.optim.lr_scheduler.StepLR,
                     "mask_type":'entmax' # "sparsemax"
                     }

    ft_model = TabNetClassifier(**tabnet_params)

    ft_model.fit(
        X_train=X_train.values, y_train=y_train.values,
        eval_set=[(X_train.values, y_train.values), (X_valid.values, y_valid.values)],
        eval_name=['train', 'valid'],
        eval_metric=['auc'],
        max_epochs=max_epochs, patience=int(max_epochs),
        batch_size=1024, virtual_batch_size=128,
        num_workers=0,
        weights=1,
        drop_last=False,
        from_unsupervised=model,
        augmentations=aug, #aug, None
    )
    return ft_model



def test_fine_tuning_tabnet(model, device, test_df, adv_examples, adv_data_before, adv_name, settings):
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
    model = fine_tune_tabnet(model, device, adv_examples, settings)
    test_data_after_ft = test_tabnet_model(model, settings, settings['TestData'])

    orig_examples, df_adv_lpf, *lpf_data_after_ft = gen_adv(model, settings, adv_name,
                                                            settings['TestData'], n=int(settings['n_test_adv']))
    lpf_data_after_ft.append(test_data_after_ft)
    mean_data_before = np.vectorize(np.mean)(np.array(adv_data_before, dtype=object))
    mean_data_after = np.vectorize(np.mean)(np.array(lpf_data_after_ft, dtype=object))
    metrics.show_finetuning_metrics(adv_data_before, lpf_data_after_ft, settings['test_string'])
    return mean_data_after - mean_data_before
