import os
from pathlib import Path

import numpy as np
import pandas as pd
import wget as wget

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

from helpers.preproccesing_tabnet_attack import get_weights, get_bounds
from preprocess_data import normalize

def prepare_data(config):
    dataset_name = config['dataset_name']
    target = config['Target']
    if dataset_name == 'credit-g':
        dataset = fetch_openml(dataset_name)

        data = pd.DataFrame(data= np.c_[dataset['data'], dataset[target]],
                            columns= dataset['feature_names'] + [target])
        data[target] = data[target].apply(lambda x : 0.0 if x == 'bad' or x == 0.0 else 1)

    else:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        out = Path(os.getcwd()+'/data/'+dataset_name+'.csv')
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            print("File already exists.")
        else:
            print("Downloading file...")
            wget.download(url, out.as_posix())
        data = pd.read_csv(out)
        data[target] = data[target].apply(lambda x : 0.0 if x == ' <=50K' or x == 0.0 else 1)


# Renaming target for training later
    config['Data'] = data

    # Normalize the data
    #scaler, train, bounds = normalize(train, target, features, bounds)
    # Compute the weights modelizing the expert's knowledge

    label_and_encode_cat_features(config)
    define_categorical_for_embedding(config)

    config['Weights'] = get_weights(data, target)
    config['Bounds'] = get_bounds(data)
    scaler, dataframe, bounds = normalize(config['Data'], target, config['FeatureNames'], config['Bounds'], config['scale_max'])
    config['Bounds'] = bounds
    config['Data'] = dataframe
    split_data(config)
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
            data.fillna(data[col].mean(), inplace=True)
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
