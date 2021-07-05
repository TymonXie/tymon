# tymon
An AI Assistant More Than a Toolkit

# Installation
    pip install tymon
# Example
## Timeseries Assistant
instant a timeseries assistant object, and choose model, set datapath.  
    
    from tymon.assistant import TimeSeries
    assistant_object = TimeSeries(model_name='LSTM',data_path='./international-airline-passengers.csv')
    assistant_object.run() 
run the code you will get a window to set the parameters for model.  
![parameters_image](/example/time_series/set_parameters.png)  
set parameters to what you want, click the start button to train the model.  
![train_process](/example/time_series/train_process.png)  
you will get final model in `./` and its performance image.  
![result_image](/example/time_series/result.png)    


