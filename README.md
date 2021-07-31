# tymon
An AI Assistant More Than a Toolkit
The reason for creating framework tymon is simple. making AI more like an assistant, helping us to complete daily AI tasks we want to accomplish, rather than just a tool. I prefer it to be interactive and "human".

# Installation
    pip install tymon
# Example
## Timeseries Assistant
### prediction for number of passengers with LSTM
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

# Related Blog
[基于tymon，无需搭建LSTM，航班人数预测](https://blog.csdn.net/tymon_xie/article/details/118501378?spm=1001.2014.3001.5502)


