import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

dataset_loc = {
    "electricity": "datasets/ECL.csv",
    "solar": "data/solar_AL.txt",
    "traffic": "data/traffic.txt",
    "exchange": "data/exchange_rate.txt",
    "weather": "data/WTH.csv",
    "belyy": "/Users/user/PycharmProjects/SensorForecasts/data/Belyy2017_GHG_zscore.csv",
    "korea": "data/korea_zscore.csv"
}

# number of variables in the time series
dataset_dims = {
    "electricity": 321,
    "weather": 12,
    "exchange": 8,
    "traffic": 862,
    "solar": 137,
    "belyy": 9,
    "korea": 25
}


class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean


def get_dataset_dims(dataset, mode):
    if mode == "single":
        return 1, 1
    elif mode == "multi":
        return dataset_dims[dataset], dataset_dims[dataset]
    else:
        print("Invalid feature mode " + mode)
        assert False


def read_data(dataset, features, seq_len, target="", scale=False, cut=None, roll=False):
    df = pd.read_csv(dataset_loc[dataset])
    df = df.drop('Unnamed: 0', axis=1)
    scaler = None

    if cut:
        end = int(len(df) * cut)
        df = df[:end]

    print(len(df))

    n_train = int(len(df) * 0.7)
    n_test = int(len(df) * 0.2)
    n_val = len(df) - (n_train + n_test)

    train_begin = 0
    train_end = n_train

    print(train_end)

    test_begin = len(df) - n_test - seq_len
    test_end = len(df)

    print(test_end - test_begin)

    val_begin = n_train - seq_len
    val_end = n_train + n_val

    print(val_end - val_begin)

    if features == "single":
        if target:
            df = df[[target]]
        else:
            df = df[df.columns[-1]]
    if features == "multi":
        if dataset in ['electricity', 'weather']:
            df = df[df.columns[1:]]
        else:
            df = df[df.columns[:]]

    if roll:
        df = df.rolling(10, axis=0).mean()
        df = df.dropna(axis=0, how='any')
    if scale:
        scaler = StandardScaler()
        train_data = df[0:n_train]
        scaler.fit(train_data.values)
        data = scaler.transform(df.values)
    else:
        data = df.values

    return data[train_begin:train_end], data[test_begin:test_end], data[val_begin:val_end], scaler, [train_begin,
                                                                                                     test_begin,
                                                                                                     val_begin], df.columns.tolist()


class seq_data(Dataset):
    def __init__(self, data, start, seq_len=20, horizon=1, args=None):
        self.data = data
        self.seq_len = seq_len
        self.horizon = horizon
        self.start = start
        self.mode = "single-step"

    def __getitem__(self, index):
        seq_begin = index
        seq_end = index + self.seq_len
        label_end = seq_end + self.horizon

        if self.mode == "single-step":
            label_begin = seq_end + self.horizon - 1

        else:
            label_begin = seq_end

        return self.data[seq_begin:seq_end], self.data[label_begin: label_end]

    def __len__(self):
        return len(self.data) - self.seq_len - self.horizon + 1


def get_dataloaders(dataset, batch_size=16, seq_len=20, horizon=1, features="single",
                    target="", scale=True, cut=None, roll=True, args=None):
    assert dataset in dataset_loc.keys()
    assert features in ["single", "multi"]
    print(dataset + " " + features)

    train, test, val, scaler, starts, col_names = read_data(dataset, features, seq_len, target, cut=cut, roll=roll)

    train_data = seq_data(train, starts[0], seq_len, horizon, args)
    test_data = seq_data(test, starts[1], seq_len, horizon, args)
    val_data = seq_data(val, starts[2], seq_len, horizon, args)

    print(train_data.data[0].shape)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader_one = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, test_loader_one, scaler, col_names
