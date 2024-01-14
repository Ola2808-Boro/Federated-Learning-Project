import os
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop,Normalize
from torch.utils.data import DataLoader,ConcatDataset,random_split
import medmnist
from medmnist import INFO
import re
import torch
from torchvision.datasets import ImageFolder
from torch import Generator
from PIL import Image
NUM_WORKERS=os.cpu_count()
from torchvision.models import resnet50
import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights,ResNet50_Weights



def prepare_img_to_predict(device,filename, model_name):
  

  data_transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  img = Image.open(f'C:/Users/olkab/Desktop/Federated Learning App/Federated-Learning-Project/server/{filename}').convert('RGB')
  input = data_transform(img)
  input = input.unsqueeze(0)
  print(f'Input shape {input.shape}')

  return input


  
def create_dataloaders_MNIST(batch_size:int,clients_number:int):
  #TODO:update desc
  """
    Description: The function is used to download data from the BreastMNIST set. 
    It takes the training and test sets, and then extracts more from the training set. 
    Returns DataLoader.

    Args:
    batch-size - how many images in one batch,
    validation_set - what percentage of the training set should be the validation set,
    num_client - the number of client in system.

    Returns:
    train_dataloaders - list of DataLoader,
    validate_dataloaders - list of DataLoader,
    test-dataloader - DataLoader,
    n_channels - number of channels,
    n_classes - number of classes.
  """

  data_transform=Compose([
      ToTensor()
  ])

  BATCH_SIZE=int(batch_size)

  download=True
  data_flag = 'breastmnist'

  info = INFO[data_flag]
  task = info['task']
  n_channels = info['n_channels']
  n_classes = len(info['label'])

  DataClass = getattr(medmnist, info['python_class'])
  print(DataClass)

  train_dataset = DataClass(split='train', transform=data_transform, download=download)
  test_dataset = DataClass(split='test', transform=data_transform, download=download)

  train_dataloader =DataLoader( train_dataset,batch_size=BATCH_SIZE,shuffle=True)
  test_dataloader=DataLoader(test_dataset,batch_size=BATCH_SIZE)
  return train_dataloader,test_dataloader,n_classes,n_channels


def create_dataloaders(batch_size:int,clients_number:int,datasets:int):


  dataset={
    'train':{
       'dataset':[],
       'dataloaders':[],
       'splited_datasets':[],
       'total_num':0,
       'num_per_item':0,

    },
    'test':{
      'dataset':[],
      'dataloaders':[],
      'splited_datasets':[],
      'total_num':0,
      'num_per_item':0,
    },
    'validation':{
      'dataset':[],
      'dataloaders':[],
      'splited_datasets':[],
      'total_num':0,
      'num_per_item':0,
    }, 

  }

  #RESNET
  data_transform = Compose([
      Resize(256),
      CenterCrop(224),
      ToTensor(),
      Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  dirs=['train_data','test_data','validation_data']
  if datasets==1:
    base_path='daj sciezke do train rsna/'
    dataset['train']['dataset'].append(ImageFolder(root=base_path+dirs[0], transform=data_transform))
    dataset['test']['dataset'].append(ImageFolder(root=base_path+dirs[1], transform=data_transform))
    dataset['validation']['dataset'].append(ImageFolder(root=base_path+dirs[2], transform=data_transform))
  elif datasets==2:
    base_path='daj sciezke do train ddsm/'
    dataset['train']['dataset'].append(ImageFolder(root=base_path+dirs[0], transform=data_transform))
    dataset['test']['dataset'].append(ImageFolder(root=base_path+dirs[1], transform=data_transform))
    dataset['validation']['dataset'].append(ImageFolder(root=base_path+dirs[2], transform=data_transform))
  elif datasets==3:
    base_path='daj sciezke do train vindir/'
    dataset['train']['dataset'].append(ImageFolder(root=base_path+dirs[0], transform=data_transform))
    dataset['test']['dataset'].append(ImageFolder(root=base_path+dirs[1], transform=data_transform))
    dataset['validation']['dataset'].append(ImageFolder(root=base_path+dirs[2], transform=data_transform))
  elif datasets==4:
    base_path_rsna='daj sciezke do train rsna/'
    base_path_ddsm='daj sciezke do train ddsm/'
    base_path_vindir='daj sciezke do train vinidr/'
    train_dataloaders=[]
    test_dataloaders=[]
    validation_dataloaders=[]
    server_test_data=[]
    data=[]
    base_paths=[base_path_rsna,base_path_ddsm,base_path_vindir]

    for base_path in base_paths:
      train_data=ImageFolder(root=base_path+dirs[0], transform=data_transform)
      test_data=ImageFolder(root=base_path+dirs[1], transform=data_transform)
      validation_data=ImageFolder(root=base_path+dirs[2], transform=data_transform)
      train_dataloaders.append(DataLoader(train_data,batch_size,shuffle=True))
      test_dataloaders.append(DataLoader(test_data,batch_size,shuffle=True))
      validation_dataloaders.append(DataLoader(validation_data,batch_size,shuffle=True))
      data.append(test_data)
    server_test_data=ConcatDataset(data)
    server_dataloaders=DataLoader(server_test_data,batch_size,shuffle=True)
    train_dataloaders.insert(0,server_dataloaders )
    test_dataloaders.insert(0,server_dataloaders )
    validation_dataloaders.insert(0,server_dataloaders )
    return train_dataloaders,test_dataloaders,validation_dataloaders
  
  else:
    print('Walk')
    for dir,sub_dir,files in os.walk('C:/Users/olkab/Desktop/Federated Learning App/Federated-Learning-Project/database'):
      #print(re.split("/cancer",dir)[0],re.split("/no_cancer",dir)[0])
      if 'no_cancer' in dir:
        dir_name=dir.replace('\\no_cancer','')
        print('Dir_name',dir_name)
        print('Result',dir.replace('\\no_cancer',''))
        if 'train' in dir:
            #dataset['train'].append(dir_name)
            print('Add ImageFolder')
            dataset['train']['dataset'].append(ImageFolder(root=dir_name, transform=data_transform))
        elif 'test' in dir_name:
            #dataset['test'].append(dir_name)
            print('Add ImageFolder')
            dataset['test']['dataset'].append(ImageFolder(root=dir_name, transform=data_transform))
        elif 'validation' in dir_name:
            #dataset['validation'].append(dir_name)
            print('Add ImageFolder')
            dataset['validation']['dataset'].append(ImageFolder(root=dir_name, transform=data_transform))
  
  #dataset['train']['dataset'].append(ImageFolder(root='C:/Users/olkab/Desktop/Federated Learning App/Federated-Learning-Project/database/vindir/test_data', transform=data_transform))
  #dataset['test']['dataset'].append(ImageFolder(root='C:/Users/olkab/Desktop/Federated Learning App/Federated-Learning-Project/database/vindir/test_data', transform=data_transform))
  #dataset['validation']['dataset'].append(ImageFolder(root='C:/Users/olkab/Desktop/Federated Learning App/Federated-Learning-Project/database/vindir/validation_data', transform=data_transform))
  if len(dataset['train']['dataset'])!=0:
    datasets=['train','test','validation']
    for dataset_name in datasets:
      print('Dataset',dataset[dataset_name]['dataset'])   
      data=ConcatDataset(dataset[dataset_name]['dataset'])
      print('Total size',data.cumulative_sizes, data,len(data))

      dataset[dataset_name]['total_num']+=len(data)
      print(f'Total num for {dataset_name},{dataset[dataset_name]["total_num"]}')
      dataset[dataset_name]["num_per_item"]=int(dataset[dataset_name]["total_num"]/(clients_number+1))
      diff=dataset[dataset_name]["total_num"]-dataset[dataset_name]["num_per_item"]*(clients_number+1)
      print(f'Diff,{diff},{dataset[dataset_name]["num_per_item"]}')
      print(dataset[dataset_name]["num_per_item"])
      generator = Generator().manual_seed(42)
      diff_data=[ dataset[dataset_name]["num_per_item"]]*(clients_number+1)
      diff_data[-1]+=diff
      print(diff_data)
      dataset[dataset_name]['splited_datasets']=random_split(data,diff_data,generator=generator)
      for data in dataset[dataset_name]['splited_datasets']:
        dataset[dataset_name]['dataloaders'].append(DataLoader(data,batch_size,shuffle=True))
    
    return dataset['train']['dataloaders'],dataset['test']['dataloaders'],dataset['validation']['dataloaders']

create_dataloaders(64,2,5)