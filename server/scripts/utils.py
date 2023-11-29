import torch
from pathlib import Path
import os
from torchsummary import summary
from mysql.connector import Error
from matplotlib.figure import Figure
import base64
from io import BytesIO

host_name='127.0.0.1'
user_name='root'
user_password='PASSWORD'
database='federated_learning'

def save_model(model:torch.nn.Module,target_dir_path:str,model_name:str,name:str):
  if Path(target_dir_path).is_dir():
    print(f'Directory {target_dir_path} already exists')
  else:
    os.makedirs(target_dir_path,exist_ok=True)
    print(f'Creating {target_dir_path} directory')

  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path +'/'+name+'_'+model_name
  print('File to save model',model_save_path)
  torch.save(obj=model.state_dict(),f=model_save_path)


def plot_charts(results,case):

    if case=='server':
        images=[]
        for index in range(len(results)):
            fig = Figure()
            ax =fig.subplots()
            ax.plot(results[index]['train']['epoch'],results[index]['train']['train_loss'],label='loss')
            ax.plot(results[index]['train']['epoch'],results[index]['train']['train_acc'],label='acc')
            ax.set_title(f'Server training - round {results[index]["round"]+1}')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Train values')
            ax.legend(loc='upper right')
            buf = BytesIO()
            fig.savefig(buf, format="png")
            data = base64.b64encode(buf.getbuffer()).decode("ascii")
            images.append(f"data:image/png;base64,{data}")
        return images
    elif case=='clients':
        images=[]
        for key in results.keys():
            for result in results[key]:
                fig = Figure()
                ax =fig.subplots()
                ax.plot(result['train']['epoch'],result['train']['train_loss'],label='loss')
                ax.plot(result['train']['epoch'],result['train']['train_acc'],label='acc')
                ax.set_title(f'Client {result["client_url"]} training - round {result["round"]+1}')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Train values')
                ax.legend(loc='upper right')
                buf = BytesIO()
                fig.savefig(buf, format="png")
                data = base64.b64encode(buf.getbuffer()).decode("ascii")
                images.append(f"data:image/png;base64,{data}")
        return images


