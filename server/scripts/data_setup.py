import os
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO


NUM_WORKERS=os.cpu_count()

def create_dataloaders_MNIST(batch_size:int):
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