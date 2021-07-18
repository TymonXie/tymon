from tymon.assistant.assistant import Assistant
from tymon.model_hub.time_series import *
from tymon.tools.para_gui import _set_paras
from .ts_data_operator import TsDataOperator
from .ts_model_operator import TsModelOperator
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


class TimeSeries(Assistant):

    def set_data_operator(self):
        data_operator = TsDataOperator(data_path=self.data_path)
        self.data_operator = data_operator

    def set_model_operator(self):
        model_operator = TsModelOperator(model_name=self.model_name)
        self.model_operator = model_operator

    def initial(self):
        # prepare workflow of data and model
        self.set_data_operator()
        self.set_model_operator()
        data_operator = self.data_operator
        model_operator = self.model_operator
        # get paras from GUI
        _set_paras(self)
        data_operator.run()
        model_operator.run()

    def train(self):
        # get operator and paras
        data_operator = self.data_operator
        model_operator = self.model_operator
        data_paras = data_operator.parameters
        model_paras = model_operator.parameters

        model = model_operator.model
        learning_rate = float(model_paras['learning_rate'])
        epoch = int(model_paras['epoch'])
        seq_length = int(model_paras['seq_length'])
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # train
        for epoch in range(epoch):
            total_loss = 0
            for iteration, X in enumerate(data_operator.trainX):  # X's shape (seq_length, n_feature)
                X = torch.tensor(X).float()
                X = torch.unsqueeze(X, 0)  # X's shape (1, seq_length, n_feature), 1 is batchsize
                output = model(X)  # output's shape (1,12)
                output = torch.squeeze(output)
                loss = loss_func(output, torch.tensor(data_operator.trainY[iteration]))
                optimizer.zero_grad()  # clear gradients for this training iteration
                loss.backward()  # computing gradients
                optimizer.step()  # update weights
                total_loss += loss

            if (epoch + 1) % 20 == 0:
                print('epoch:{:3d}, loss:{:6.4f}'.format(epoch + 1, total_loss.data.numpy()))
        torch.save({'state_dict': model.state_dict()}, 'checkpoint.pth.tar')


    def eval(self):
        # get operator and paras
        data_operator = self.data_operator
        model_operator = self.model_operator
        data_paras = data_operator.parameters
        model_paras = model_operator.parameters
        seq_length = int(model_paras['seq_length'])

        model = model_operator.model
        checkpoint = torch.load('checkpoint.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        # predict
        model.eval()
        eval_result = []
        train_legth = len(data_operator.trainData)
        test_length = len(data_operator.testData)
        latest_data = data_operator.trainData[train_legth - seq_length:train_legth, :]
        latest_data = list(latest_data)
        i = 0
        while i < test_length:
            x = torch.tensor(np.array(latest_data))
            x = torch.unsqueeze(x, 0)
            output = model(x)
            output = torch.squeeze(output)
            eval_result.append(output.data.numpy())
            latest_data.append(output.data.numpy())
            latest_data.remove(latest_data[0])
            i += 1
        plt.figure()
        eval_result = np.array(eval_result)
        eval_result = eval_result.reshape(-1, 1).squeeze()

        all_data_length = len(data_operator.sourceData)
        source_data = data_operator.sourceData.reshape(-1, 1).squeeze()
        plt.plot(list(range(all_data_length)), source_data, label='source data')

        eval_length = len(eval_result)
        plt.plot(list(range(all_data_length - eval_length, all_data_length)), eval_result,
                 label='predict data')

        plt.legend(loc='best')
        plt.show()

    def train_and_eval(self):
        self.train()
        self.eval()
