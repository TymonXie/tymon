from tymon.assistant.data_operator import DataOperator
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data_paras = {
    'n_feature': 12,
    'seq_length': 4,
    'divide_ratio': 0.7
}


class TsDataOperator(DataOperator):
    def __init__(self, data_path):
        self.data_path = data_path
        self.parameters = data_paras

        self.sourceData = None
        self.trainData = None
        self.testData = None
        self.trainX = None
        self.trainY = None

    def data_read_and_preprocess(self):
        data = pd.read_csv(self.data_path)
        data = data.iloc[:, 1:5].values
        data = np.array(data).astype(np.float32)
        sc = MinMaxScaler()
        data = sc.fit_transform(data)  # 归一化
        self.sourceData = data

    def feature_engineering(self):
        pass

    def data_divide(self):
        n_feature = int(self.parameters['n_feature'])
        divide_ratio = float(self.parameters['divide_ratio'])
        data = self.sourceData.reshape(-1, n_feature)
        train_source = data[:int(len(data) * divide_ratio), :]
        test_source = data[int(len(data) * divide_ratio):, :]
        self.trainData = train_source
        self.testData = test_source

    def data_generator(self):
        # generate train data
        train_data_x = []
        train_data_y = []
        seq_length = int(self.parameters['seq_length'])
        for i in range(self.trainData.shape[0] - seq_length):
            temp_x = self.trainData[i:i + seq_length, :]
            temp_y = self.trainData[i + seq_length, :]
            train_data_x.append(temp_x)
            train_data_y.append(temp_y)
        self.trainX = train_data_x
        self.trainY = train_data_y

    def run(self):
        self.data_read_and_preprocess()
        self.data_divide()
        self.data_generator()
