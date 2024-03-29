import keras
import torch
import numpy as np
from torch.utils.data import Dataset


from keras.utils import np_utils


class TabularDataset(Dataset):
    def __init__(self, dataframe, features, target, is_train, transform=None, target_transform=None):
        self.seed = 0
        self.dataset_name = 'credit-g'
        self.is_train = is_train
        self.dataframe = dataframe
        self.target = target
        self.features = features
        self.features_tensor = torch.DoubleTensor(dataframe[features].values)
        n_classes = len(np.unique(dataframe[target]))
        self.target_tensor = torch.DoubleTensor(keras.utils.np_utils.to_categorical(dataframe[target].values, num_classes=2))

        self.features_dim = self.features_tensor.size(dim=1)
        self.target_dim = self.target_tensor.size(dim=1)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        data = self.features_tensor[index]
        label = self.target_tensor[index]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label

    def get_dimensions(self):
        return self.features_dim, self.target_dim