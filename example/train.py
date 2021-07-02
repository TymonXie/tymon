from tymon.assistant import TimeSeries

assistant_object = TimeSeries(model_name='LSTM',data_path='./international-airline-passengers.csv')
assistant_object.run()
