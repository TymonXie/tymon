class Assistant():
    def __init__(self, data_path, model_name):
        self.data_path = data_path
        self.model_name = model_name

        # personal values
        self._model = None
        self._sourceData = None
        self._parameters = None
        self._trainData = None
        self._testData = None
        self._trainX = None
        self._trainY = None


    def _data_divide(self):
        pass

    def _data_generator(self):
        pass

    def model(self):
        pass

    def _select_model(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def predict(self):
        pass

    def run(self):
        pass
