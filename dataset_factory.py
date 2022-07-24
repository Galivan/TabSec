from sklearn.model_selection import train_test_split

from tabular_dataset import TabularDataset
import fetch_data


def get_train_test_dataset(dataset_name, test_size=None, train_size=None, seed=0, transform=None, target_transform=None):
    dataframe, target, features = fetch_data.get_df(dataset_name)
    train_df, test_df = train_test_split(dataframe, test_size=test_size, train_size=train_size,
                                         random_state=seed, shuffle=True)
    train_dataset = TabularDataset(train_df, target, True, transform, target_transform)
    test_dataset = TabularDataset(test_df, target, False, transform, target_transform)
    return train_dataset, test_dataset
