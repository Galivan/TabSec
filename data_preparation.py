import torch
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

from helpers.preproccesing_tabnet_attack import get_weights, normalize, get_bounds

def prepare_data(config):
    dataset_name = config['dataset_name']
    target = config['Target']

    print(f'target is {target}')
     
    dataset = fetch_openml(dataset_name)

    data = pd.DataFrame(data= np.c_[dataset['data'], dataset[target]],
                    columns= dataset['feature_names'] + [target]) 

    # Renaming target for training later
    data[target] = data[target].apply(lambda x : 0.0 if x == 'bad' or x == 0.0 else 1)

    config['Data'] = data

    # Normalize the data
    #scaler, train, bounds = normalize(train, target, features, bounds)
    # Compute the weights modelizing the expert's knowledge

    label_and_encode_cat_features(config)
    define_categorical_for_embedding(config)

    split_data(config)
    config['Weights'] = get_weights(data, target)
    config['Bounds'] = get_bounds(data)

    return config

def split_data(config):
    train = config['Data']
    train_data, test_data = train_test_split(train, test_size=0.3, stratify=train[config['Target']])
    test_data, validation_data = train_test_split(test_data, test_size=0.5, stratify=test_data[config['Target']])

    config['TrainData'] = train_data
    config['TestData'] = test_data
    config['ValidData'] = validation_data

def label_and_encode_cat_features(config):
    data = config['Data']

    nunique = data.nunique()
    types = data.dtypes

    categorical_columns = []
    categorical_dims =  {}
    for col in data.columns:
        if types[col] == 'object' or nunique[col] < 200:
            print(col, data[col].nunique())
            l_enc = LabelEncoder()
            data[col] = data[col].fillna("VV_likely")
            data[col] = l_enc.fit_transform(data[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        else:
            data.fillna(data.loc[col].mean(), inplace=True)
    config['categorical_columns'] = categorical_columns
    config['categorical_dims'] = categorical_dims

def define_categorical_for_embedding(config):
    # unused_feat = ['Set']
    data = config['Data']
    target = config['Target']
    categorical_columns = config['categorical_columns']
    categorical_dims = config['categorical_dims']
    
    unused_feat = []

    features = [ col for col in data.columns if col not in unused_feat+[target]]

    cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

    cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    config['FeatureNames'] = features
    config['cat_idxs'] = cat_idxs
    config['cat_dims'] = cat_dims
