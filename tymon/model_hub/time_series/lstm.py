import torch.nn as nn


# model
class LSTM(nn.Module):
    def __init__(self, in_dim=12, hidden_dim=10, output_dim=12, n_layer=1):
        super(LSTM, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layer = n_layer
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=n_layer, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h_out, _) = self.lstm(x)  # h_out是序列最后一个元素的hidden state
        h_out = h_out.view(h_out.shape[0], -1)
        h_out = self.linear(h_out)
        return h_out

lstm_para = {'epoch': 700,
             'learning_rate': 0.001,
             'seq_length': 4,
             'n_feature': 12,
             'divide_ratio': 0.7
             }
