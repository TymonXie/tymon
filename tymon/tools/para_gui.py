import tkinter


def _set_paras(assistant):
    model_name = assistant._model._get_name()
    parameters = assistant._parameters
    print('default parameters:',parameters)
    entry_list = []
    root = tkinter.Tk()
    root.title(model_name)
    index = 0
    for key,value in parameters.items():
        label = tkinter.Label(root, text=key +" : ")
        label.grid(row = index)
        entry = tkinter.Entry(root)
        entry.insert(0,str(value))
        entry.grid(row=index, column=1)
        entry_list.append(entry)
        index +=1

    def setValues():
        index = 0
        for key,_ in parameters.items():
            assistant._parameters[key] = entry_list[index].get()
            index +=1

        root.destroy()

    buttonStart = tkinter.Button(root, text="start", command=setValues)
    buttonStart.grid(row=index, sticky=tkinter.W, padx=20)

    root.mainloop()