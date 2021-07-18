from tymon.assistant.model_operator import ModelOperator
from tymon.model_hub.time_series import *


class TsModelOperator(ModelOperator):

    def model_selector(self):
        if self.model_name == 'LSTM':
            model = LSTM()
            parameters = lstm_paras
        self.model = model
        self.parameters = parameters

    def run(self):
        pass
