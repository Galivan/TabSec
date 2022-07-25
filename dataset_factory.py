import fetch_data
from sklearn.model_selection import train_test_split

from tabular_dataset import TabularDataset
from preprocess_data import get_bounds, get_weights



def get_train_test_dataset(config, dataset_name, test_size=None, train_size=None, seed=0, transform=None, target_transform=None):
    dataframe, target, features = fetch_data.get_df(dataset_name)
    train_df, test_df = train_test_split(dataframe, test_size=test_size, train_size=train_size,
                                         random_state=seed, shuffle=True)
    # configurations needed for adversarial generation
    weights = get_weights(dataframe, target)
    bounds = get_bounds(dataframe)

    config['TrainData'] = train_df
    config['TestData'] = test_df
    config['Target'] = target
    config['FeatureNames'] = features
    config['Weights'] = weights
    config['Bounds'] = bounds 

    train_dataset = TabularDataset(train_df,features, target, True, transform, target_transform)
    test_dataset = TabularDataset(test_df,features, target, False, transform, target_transform)
    return train_dataset, test_dataset

