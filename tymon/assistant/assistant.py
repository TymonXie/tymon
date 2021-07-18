class Assistant():
    def __init__(self, data_path, model_name):
        self.data_path = data_path
        self.model_name = model_name

        # operators
        self.data_operator = None
        self.model_operator = None

        # initial all operatros
        self.initial()

    def set_data_operator(self):
        pass

    def set_model_operator(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def train_and_eval(self):
        pass

    def predict(self):
        pass

    def initial(self):
        pass
