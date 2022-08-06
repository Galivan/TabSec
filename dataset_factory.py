import fetch_data
from sklearn.model_selection import train_test_split

from tabular_dataset import TabularDataset
from preprocess_data import get_bounds, get_weights, normalize


def get_train_test_dataset(config, dataset_name, test_size=None, train_size=None, seed=0, transform=None, target_transform=None):
    dataframe, target, features = fetch_data.get_df(dataset_name)

    # Compute the bounds for clipping
    bounds = get_bounds(dataframe)
    # Normalize the data
    scaler, dataframe, bounds = normalize(dataframe, target, features, bounds)

    # Compute the weights modelizing the expert's knowledge
    weights = get_weights(dataframe, target)
    train_df, test_df = train_test_split(dataframe, test_size=test_size, train_size=train_size,
                                         random_state=seed, shuffle=True)
    



    train_df, test_df = train_test_split(dataframe, test_size=test_size, train_size=train_size,
                                         random_state=seed, shuffle=True)
    config['TrainData'] = train_df
    config['TestData'] = test_df
    config['Target'] = target
    config['FeatureNames'] = features
    config['Weights'] = weights
    config['Bounds'] = bounds 

    train_dataset = TabularDataset(train_df,features, target, True, transform, target_transform)
    test_dataset = TabularDataset(test_df,features, target, False, transform, target_transform)
    return train_dataset, test_dataset

