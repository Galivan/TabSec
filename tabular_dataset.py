from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, dataframe, target, is_train, transform=None, target_transform=None):
        self.seed = 0
        self.dataset_name = 'credit-g'
        self.is_train = is_train
        self.dataframe = dataframe
        self.target = target
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        data = self.dataframe.iloc[index]
        label = data[self.target]
        data.drop(self.target, inplace=True)
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label