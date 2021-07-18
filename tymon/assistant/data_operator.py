class DataOperator:
    def __init__(self, data_path):
        self.data_path = data_path
        self.parameters = None

    def data_read_and_preprocess(self):
        # read different type of datasource and do preprocess
        pass

    def feature_engineering(self):
        # choose the best feature
        pass

    def data_divide(self):
        pass

    def data_generator(self):
        # generate data for training
        pass

    def run(self):
        # define data workflow
        pass
