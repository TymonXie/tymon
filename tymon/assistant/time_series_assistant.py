from .assistant import Assistant
from tymon.model_hub.time_series import *
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

# TODO: put parameters into other station
# super parameters
EPOCH = 100
learning_rate = 0.01
seq_length = 4  # 序列长度
n_feature = 12
divide_ratio = 0.6


class TimeSeries(Assistant):
    def data_generator(self):
        data = pd.read_csv(self.data_path)
        data = data.iloc[:, 1:5].values
        data = np.array(data).astype(np.float32)
        sc = MinMaxScaler()
        data = sc.fit_transform(data)  # 归一化

        # TODO: divide data function
        train_source = data[:int(len(data) * divide_ratio), :]
        test_source = data[:int(len(data) * divide_ratio), :]

        # generate train data
        train_data_x = []
        train_data_y = []
        for i in range(data.shape[0] - seq_length):
            temp_x = data[i:i + seq_length, :]
            temp_y = data[i + seq_length, :]
            train_data_x.append(temp_x)
            train_data_y.append(temp_y)
        return train_data_x, train_data_y, test_source

    def train(self):
        if self.model_name == 'LSTM':
            model = LSTM()


