import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class data_detime(Dataset):
    def __init__(self, data, lookback_length, lookforward_length, multi_steps=False):
        self.seq_len = lookback_length
        self.pred_len = lookforward_length
        self.multi_steps = multi_steps
        self.data_y = data
        self.data=data
        print(self.data.shape)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        x = self.data[s_begin:s_end]
        # x_else = self.data_else[s_begin:s_end]
        if self.multi_steps:
            y = self.data_y[s_end:s_end + self.pred_len, 0]
        else:
            y = self.data_y[s_end + self.pred_len - 1:s_end + self.pred_len, 0]
        return x, y

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

def split_data_cnn(data, train, test, lookback_length):
    # data = data.loc[~(data['gen'] == 0)]
    for column in list(data.columns[data.isnull().sum() > 0]):
        data[column].interpolate(method='linear', limit_direction='forward')
    timestamp = data[['date']]
    timestamp['date'] = pd.to_datetime(timestamp.date)
    cols = list(data.columns)
    cols.remove('date')
    data = data[cols].values
    data[:, 0] = np.maximum(data[:, 0], 0)
    length = len(data)
    num_train = int(length * train)
    num_test = int(length * test)
    num_valid = length - num_test - num_train

    timestamp_train = timestamp[0:num_train]
    timestamp_valid = timestamp[num_train - lookback_length:num_train + num_valid]
    timestamp_test = timestamp[num_train + num_valid - lookback_length:]
    scalar = StandardScaler()
    # scalar = MinMaxScaler()
    scalar_y = StandardScaler()
    # scalar_y = MinMaxScaler()
    y = data[0:num_train, 0].reshape(-1, 1)
    scalar_y.fit(y)
    # y_trans=scalar_y.transform(y)
    # y_re=scalar_y.inverse_transform(y_trans.reshape(-1,1))
    # scalar = MinMaxScaler()
    scalar.fit(data[0:num_train])
    data = scalar.transform(data)
    # data_re=scalar.inverse_transform(data)
    data_train = data[0:num_train]
    data_valid = data[num_train - lookback_length:num_train + num_valid]
    data_test = data[num_train + num_valid - lookback_length:length]

    # plt.plot(range(num_train), data[0:num_train, 0], 'r', range(num_train, num_train + num_valid), data[num_train:num_train + num_valid, 0], 'g',
    #          range(num_train + num_valid, length), data[num_train + num_valid:length, 0], 'b')
    # plt.title(f'{site}-{dataset}')
    # plt.show()
    return data_train, data_valid, data_test, timestamp_train, timestamp_valid, timestamp_test, scalar_y








