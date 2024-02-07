
import torch
import glob
import os


def fedAvgAlgorithm(model):
    """
    Description: The function implements FedAvg federated learning strategies.

    Args:
    model - model
    """
    weights={}
    bias={} 
    #test=[]
    clients=0
    for file in glob.glob('Federated-Learning-Project/server/data/client/*'):
        clients+=1
        print(file)
        for name, param in model.named_parameters(): 
            print(name)
            if 'bias' in name:
                if name in bias.keys():
                    print(f'Name in bias already exists',bias.keys(),name)
                    bias[name]+=param.data
                else:
                    print(f'Create key',bias.keys(),name)
                    bias[name]=param.data
            else:
                if name in weights.keys():
                    print(f'Name in bias already exists',weights.keys(),name)
                    weights[name]+=param.data
                else:
                    print(f'Create key',weights.keys(),name)
                    weights[name]=param.data

    # print(bias['fc.4.bias'])
    for key in weights.keys():
        print(key)
        weights[key]=weights[key]/(clients)
    for key in bias.keys():
        print(key)
        bias[key]=bias[key]/(clients)


    save_new_param(model=model,weights=weights,bias=bias)



def save_new_param(model,weights,bias):
    """
    Description: Function used to save model weights.

    Args:
    model - model
    weights- weights after AvgFed algorithm
    bias - bias
    """
    model_new=model
    file_save='data/client_for_server.pt'
    for name, param in model_new.named_parameters():
        if 'bias' in name:
            for key in bias.keys():
                model_new.name=bias[key]
        elif 'weight' in name:
            for key in weights.keys():
                model_new.name=weights[key]


    torch.save(obj=model_new.state_dict(),f=file_save)


