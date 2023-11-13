import torch
from pathlib import Path
import os

def save_model(model:torch.nn.Module,target_dir_path:str,model_name:str):
  if Path(target_dir_path).is_dir():
    print(f'Directory {target_dir_path} already exists')
  else:
    os.makedirs(target_dir_path,exist_ok=True)
    print(f'Creating {target_dir_path} directory')

  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path +'/'+ model_name

  torch.save(obj=model.state_dict(),f=model_save_path)