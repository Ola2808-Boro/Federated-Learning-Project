

import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

def choose_model(model:str):

    if model.lower()=='resnet18':
        print('Model ',model)
        model=resnet18(ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(512, 1)
        return model


