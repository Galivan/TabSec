from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import fetch_data

from tabular_dataset import TabularDataset
from preprocess_data import get_bounds, get_weights, normalize


def get_train_test_dataset(settings, dataset_name, test_size=None, train_size=None, seed=0, transform=None, target_transform=None):
    dataframe, target, features = fetch_data.get_df(dataset_name)

    # Compute the bounds for clipping
    bounds = get_bounds(dataframe)

    # Normalize the data
    scaler, dataframe, bounds = normalize(dataframe, target, features, bounds)

    # Compute the weights modelizing the expert's knowledge
    weights = get_weights(dataframe, target)

    train_df, test_df = train_test_split(dataframe, test_size=test_size, train_size=train_size,
                                         random_state=seed, shuffle=True)

    settings['TrainData'] = train_df
    settings['TestData'] = test_df
    settings['Target'] = target
    settings['FeatureNames'] = features
    settings['Weights'] = weights
    settings['Bounds'] = bounds

    train_dataset = TabularDataset(train_df,features, target, True, transform, target_transform)
    test_dataset = TabularDataset(test_df,features, target, False, transform, target_transform)
    return train_dataset, test_dataset


def get_credit_g_dataloaders(settings):
    credit_g = 'credit-g'
    credit_g_train, credit_g_test = get_train_test_dataset(settings, credit_g, test_size=300)


    train_dataloader = DataLoader(credit_g_train, batch_size=settings['batch_size'], shuffle=True)
    test_dataloader = DataLoader(credit_g_test, batch_size=len(credit_g_test))
    d_in, d_out = credit_g_train.get_dimensions()
    return train_dataloader, test_dataloader, (d_in, d_out)
