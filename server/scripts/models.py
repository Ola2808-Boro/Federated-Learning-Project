

import torch
from torch import nn
from torchvision.models import resnet18, resnet50,efficientnet_b0
from torchvision.models import ResNet18_Weights, ResNet50_Weights, EfficientNet_B0_Weights

def choose_model(model:str):

    # path=''
    # if model.lower()=='resnet50-vindir':
    #     print('Model ',model)
    #     path=''
    # elif model.lower()=='resnet50-rsna':
    #     print('Model',model)
    #     path=''
    # elif model.lower()=='resnet50-ddsm':
    #     print('Model',model)
    #     path=''
    # elif model.lower()=='efficientNet-vindir':
    #     print('Model',model)
    #     path=''
    # elif model.lower()=='efficientNet-rsna':
    #     print('Model',model)
    #     path=''
    # elif model.lower()=='efficientNet-ddsm':
    #     print('Model',model)
    #     path=''
    # elif model.lower()=='adda-resnet50-vindir-rsna':
    #     print('Model',model)
    #     path=''
    # elif model.lower()=='adda-resnet50-vindir-ddsm':
    #     print('Model',model)
    #     path=''
    # elif model.lower()=='reverse-gradient-resnet50-vindir-rsna':
    #     print('Model',model)
    #     path=''
    # elif model.lower()=='reverse-gradient-resnet50-vindir-ddsm':
    #     print('Model',model)
    #     path=''

    path='C:/Users/olkab\Desktop/Federated Learning App/Federated-Learning-Project/server/models/resnet50new.pt'
    if 'resnet' in model:
        resnet = resnet50(ResNet50_Weights.DEFAULT)
        num_features = resnet.fc.in_features
        resnet.fc = nn.Sequential(
        nn.Linear(num_features, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Dropout(p=0.6),
        nn.Linear(16, 2)
        )
        model = resnet
        model = torch.load(path)
    elif 'efficientNet' in model:
        efficientNet=efficientnet_b0(EfficientNet_B0_Weights.DEFAULT)
        num_features = efficientNet.fc.in_features
        efficientNet.fc = nn.Sequential(
        nn.Linear(num_features, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Dropout(p=0.6),
        nn.Linear(16, 2)
        )
        model = efficientNet
        model = torch.load(path)

    return model
       


choose_model(model='resnet18')