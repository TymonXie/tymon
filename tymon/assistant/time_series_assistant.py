from .assistant import Assistant
from tymon.model_hub.time_series import *
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

# TODO: put parameters into other station
# super parameters
EPOCH = 700
learning_rate = 0.001
seq_length = 4  # 序列长度
n_feature = 12
divide_ratio = 0.7


class TimeSeries(Assistant):

    def _data_divide(self):
        data = pd.read_csv(self.data_path)
        data = data.iloc[:, 1:5].values
        data = np.array(data).astype(np.float32)
        sc = MinMaxScaler()
        data = sc.fit_transform(data)  # 归一化
        self._sourceData = data
        data = data.reshape(-1, n_feature)
        # TODO: divide data function
        train_source = data[:int(len(data) * divide_ratio), :]
        test_source = data[int(len(data) * divide_ratio):, :]
        self._trainData = train_source
        self._testData = test_source

    def _data_generator(self):
        # generate train data
        train_data_x = []
        train_data_y = []
        for i in range(self._trainData.shape[0] - seq_length):
            temp_x = self._trainData[i:i + seq_length, :]
            temp_y = self._trainData[i + seq_length, :]
            train_data_x.append(temp_x)
            train_data_y.append(temp_y)
        self._trainX = train_data_x
        self._trainY = train_data_y


    def _select_model(self):
        if self.model_name == 'LSTM':
            model = LSTM()
        self._model = model

    def run(self):
        model = self._model
        train = True
        if train:
            loss_func = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            # train
            for epoch in range(EPOCH):
                total_loss = 0
                for iteration, X in enumerate(self._trainX):  # X's shape (seq_length, n_feature)
                    X = torch.tensor(X).float()
                    X = torch.unsqueeze(X, 0)  # X's shape (1, seq_length, n_feature), 1 is batchsize
                    output = model(X)  # output's shape (1,12)
                    output = torch.squeeze(output)
                    loss = loss_func(output, torch.tensor(self._trainY[iteration]))
                    optimizer.zero_grad()  # clear gradients for this training iteration
                    loss.backward()  # computing gradients
                    optimizer.step()  # update weights
                    total_loss += loss

                if (epoch + 1) % 20 == 0:
                    print('epoch:{:3d}, loss:{:6.4f}'.format(epoch + 1, total_loss.data.numpy()))
            torch.save({'state_dict': model.state_dict()}, 'checkpoint.pth.tar')
        else:
            checkpoint = torch.load('checkpoint.pth.tar')
            model.load_state_dict(checkpoint['state_dict'])

        # predict
        model.eval()
        eval_result = []
        train_legth = len(self._trainData)
        test_length = len(self._testData)
        latest_data = self._trainData[train_legth-seq_length:train_legth,:]
        latest_data = list(latest_data)
        i = 0
        while i < test_length:
            x = torch.tensor(np.array(latest_data))
            x = torch.unsqueeze(x,0)
            output = model(x)
            output = torch.squeeze(output)
            eval_result.append(output.data.numpy())
            latest_data.append(output.data.numpy())
            latest_data.remove(latest_data[0])
            i+=1
        plt.figure()
        eval_result = np.array(eval_result)
        eval_result = eval_result.reshape(-1,1).squeeze()

        all_data_length = len(self._sourceData)
        source_data = self._sourceData.reshape(-1,1).squeeze()
        plt.plot(list(range(all_data_length)),source_data,label= 'source data')

        eval_length = len(eval_result)
        plt.plot(list(range(all_data_length-eval_length,all_data_length)),eval_result,
                 label = 'predict data')



        plt.legend(loc = 'best')
        plt.show()













