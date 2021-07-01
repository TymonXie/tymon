from tymon.assistant import TimeSeries

assistant_object = TimeSeries(model_name='LSTM',data_path='./international-airline-passengers.csv')
train_x,train_y,test_data = assistant_object.data_generator()
print(train_x)