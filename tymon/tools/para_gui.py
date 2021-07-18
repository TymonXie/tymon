import tkinter


def _set_paras(assistant):
    data_operator = assistant.data_operator
    model_operator = assistant.model_operator

    data_parameters = data_operator.parameters
    model_parameters = model_operator.parameters

    print('default data parameters:',data_parameters)
    print('default model parameters:',model_parameters)

    entry_list_data = []
    entry_list_model = []
    root = tkinter.Tk()
    root.title(assistant.model_name)
    index_data_paras = 0
    label = tkinter.Label(root, text='data parameters')
    label.grid(row=0)
    index_data_paras+=1

    # set dataparas inputs
    for key,value in data_parameters.items():
        label = tkinter.Label(root, text=key +" : ")
        label.grid(row = index_data_paras)
        entry = tkinter.Entry(root)
        entry.insert(0,str(value))
        entry.grid(row=index_data_paras, column=1)
        entry_list_data.append(entry)
        index_data_paras +=1

    # set modelparas inputs
    index_model_paras = index_data_paras
    label = tkinter.Label(root, text='model_parameters')
    label.grid(row=index_model_paras)
    index_model_paras +=1
    for key,value in model_parameters.items():
        label = tkinter.Label(root, text=key +" : ")
        label.grid(row = index_model_paras)
        entry = tkinter.Entry(root)
        entry.insert(0,str(value))
        entry.grid(row=index_model_paras, column=1)
        entry_list_model.append(entry)
        index_model_paras +=1

    def setValues():
        index = 0
        for key,_ in data_parameters.items():
            assistant.data_operator.parameters[key] = entry_list_data[index].get()
            index +=1
        index = 0
        for key,_ in model_parameters.items():
            assistant.model_operator.parameters[key] = entry_list_model[index].get()
            index +=1

        root.destroy()

    final_index = max(index_data_paras,index_model_paras)
    buttonStart = tkinter.Button(root, text="start", command=setValues)
    buttonStart.grid(row=final_index, sticky=tkinter.W, padx=20)

    root.mainloop()