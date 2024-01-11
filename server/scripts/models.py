

import torch
from torch import nn
from torchvision.models import resnet18, resnet50,efficientnet_b0
from torchvision.models import ResNet18_Weights, ResNet50_Weights, EfficientNet_B0_Weights

def choose_model(model_name:str):

    path=''
    model=resnet50(ResNet50_Weights.DEFAULT)
    if model_name.lower()=='resnet50-vindir':
        print('Model resnet50-vindir',model)
        path='C:/Users/olkab/Desktop/Federated Learning App/Federated-Learning-Project/server/models/resnet50new.pt'
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
        model.load_state_dict(torch.load(path))
    elif model_name.lower()=='resnet50-rsna':
        print('Model',model)
        path=''
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
        model.load_state_dict(torch.load(path))
    elif model_name.lower()=='resnet50-ddsm':
        print('Model',model)
        path=''
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
        model.load_state_dict(torch.load(path))
    elif model_name.lower()=='efficientNet-vindir':
        print('Model',model)
        path=''
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
        model.load_state_dict(torch.load(path))
    elif model_name.lower()=='efficientNet-rsna':
        print('Model',model)
        path=''
        num_features = efficientNet.fc.in_features
        efficientNet.fc = nn.Sequential(
        nn.Linear(num_features, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Dropout(p=0.6),
        nn.Linear(16, 2)
        )
        model = efficientNet
        model.load_state_dict(torch.load(path))
    elif model_name.lower()=='efficientNet-ddsm':
        print('Model',model)
        path=''
    elif model_name.lower()=='adda-resnet50-vindir-rsna':
        print('Model',model)
        path=''
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
    elif model_name.lower()=='adda-resnet50-vindir-ddsm':
        print('Model',model)
        path=''
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
    elif model_name.lower()=='reverse-gradient-resnet50-vindir-rsna':
        print('Model',model)
        path=''
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
    elif model_name.lower()=='reverse-gradient-resnet50-vindir-ddsm':
        print('Model',model)
        path=''
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

    return model
       


