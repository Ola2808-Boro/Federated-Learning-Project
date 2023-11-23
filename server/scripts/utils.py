import torch
from pathlib import Path
import os
import sqlite3
from torchsummary import summary
import re
import mysql.connector
from mysql.connector import Error

host_name='127.0.0.1'
user_name='root'
user_password='PASSWORD'
database='federated_learning'

def save_model(model:torch.nn.Module,target_dir_path:str,model_name:str):
  save_to_database()
  if Path(target_dir_path).is_dir():
    print(f'Directory {target_dir_path} already exists')
  else:
    os.makedirs(target_dir_path,exist_ok=True)
    print(f'Creating {target_dir_path} directory')

  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path +'/'+ model_name
  print('File to save model',model_save_path)
  with open('Federated-Learning-Project/server/data/model_net.txt','a') as f:
            print('Open file')
            f.write(f'{model.state_dict() } \n')
  with open(file='Federated-Learning-Project/server/data/paramas.txt',mode='a',encoding="utf-8") as f:
            f.write(f'{summary(model,(1,28,28)) } \n')
  with open(file='Federated-Learning-Project/server/data/paramas_ola.txt',mode='w',encoding="utf-8") as f:
            for parameter in model.parameters():
              f.write(f'{parameter.flatten()} \n')
        


  torch.save(obj=model.state_dict(),f=model_save_path)


