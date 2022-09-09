import torch
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml

from helpers.preproccesing_tabnet_attack import get_weights, normalize, get_bounds

def prepare_data(config):
    dataset_name = config['dataset_name']
    target = config['Target']

    print(f'target is {target}')
     
    dataset = fetch_openml(dataset_name)

    train = pd.DataFrame(data= np.c_[dataset['data'], dataset[target]],
                    columns= dataset['feature_names'] + [target]) 

    # Renaming target for training later
    train[target] = train[target].apply(lambda x : 0.0 if x == 'bad' or x == 0.0 else 1)

    config['train'] = train

    # Normalize the data
    #scaler, train, bounds = normalize(train, target, features, bounds)
    # Compute the weights modelizing the expert's knowledge

    label_and_encode_cat_features(config)
    define_categorical_for_embedding(config)

    split_data(config)
    config['Weights'] = get_weights(train, target)
    config['Bounds'] = get_bounds(train)

    return config

def split_data(config):
    train = config['train']
    train_set = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(train.shape[0],))

    train_indices = train[train_set=="train"].index
    valid_indices = train[train_set=="valid"].index
    test_indices = train[train_set=="test"].index

    config['train_indices'] = train_indices
    config['valid_indices'] = valid_indices
    config['test_indices'] = test_indices
    config['TestData'] = train.iloc[test_indices]


def label_and_encode_cat_features(config):
    train = config['train']

    nunique = train.nunique()
    types = train.dtypes

    categorical_columns = []
    categorical_dims =  {}
    for col in train.columns:
        if types[col] == 'object' or nunique[col] < 200:
            print(col, train[col].nunique())
            l_enc = LabelEncoder()
            train[col] = train[col].fillna("VV_likely")
            train[col] = l_enc.fit_transform(train[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        else:
            train.fillna(train.loc[col].mean(), inplace=True)
    config['categorical_columns'] = categorical_columns
    config['categorical_dims'] = categorical_dims

def define_categorical_for_embedding(config):
    # unused_feat = ['Set']
    train = config['train']
    target = config['Target']
    categorical_columns = config['categorical_columns']
    categorical_dims = config['categorical_dims']
    
    unused_feat = []

    features = [ col for col in train.columns if col not in unused_feat+[target]] 

    cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

    cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    config['FeatureNames'] = features
    config['cat_idxs'] = cat_idxs
    config['cat_dims'] = cat_dims
