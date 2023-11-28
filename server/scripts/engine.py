import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import os

accuracy = Accuracy(task="binary", num_classes=2)


def train_step(model:nn.Module, dataloader:DataLoader,optimizer:torch.optim.Optimizer,loss_fn:nn.Module,device:torch.device):

  model.train()
  loss_avg=0
  acc_avg=0
  for batch,(x,y) in enumerate(dataloader):
    x,y=x.to(device),y.to(device)
    y_pred=model(x)
    y = y.squeeze().long()
    #y_pred_class=torch.round(torch.sigmoid(y_pred)) #to use to measure accuracy
    y_pred_class=torch.softmax(y_pred,dim=1).argmax(dim=1)
    #print(y.shape,y_pred.shape)
    loss=loss_fn(y_pred,y)
    loss_avg=loss_avg+loss.item()
    acc=accuracy(y_pred_class,y)
    acc_avg=acc_avg+acc
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  loss_avg=loss_avg/len(dataloader)
  acc_avg=acc_avg/len(dataloader)
  return loss_avg,acc_avg.item()

def test_step(model:nn.Module, dataloader:DataLoader,loss_fn:nn.Module,device:torch.device):
  model.eval()
  with torch.inference_mode():
    loss_avg=0
    acc_avg=0
    for batch,(x,y) in enumerate(dataloader):
      x,y=x.to(device),y.to(device)
      y_pred=model(x)
      y = y.squeeze().long()
      #y_pred_class=torch.round(torch.sigmoid(y_pred)) #to use to measure accuracy
      y_pred_class=torch.softmax(y_pred,dim=1).argmax(dim=1) #to use to measure accuracy
      #print(y.shape,y_pred.shape)
      loss=loss_fn(y_pred,y)
      loss_avg=loss_avg+loss.item()
      acc=accuracy(y_pred_class,y)
      acc_avg=acc_avg+acc
    loss_avg=loss_avg/len(dataloader)
    acc_avg=acc_avg/len(dataloader)
    return loss_avg,acc_avg.item()

def train(model:nn.Module, train_dataloader:DataLoader,test_dataloader:DataLoader,epochs:int,optimizer:str,lr:float,case:str):

  if os.path.isfile('Federated-Learning-Project/server/data/server_for_client_net'):
    if case=='server':
      model.load_state_dict(torch.load('Federated-Learning-Project/server/data/client_for_server_net.pt'))
    elif case=='client':
      model.load_state_dict(torch.load('Federated-Learning-Project/server/data/server_for_client_net.pt'))
  loss_fn= torch.nn.CrossEntropyLoss()
  device='cuda' if torch.cuda.is_available() else 'cpu'
  result_train={
      'epoch':[],
      'train_loss':[],
      'train_acc':[],
  }

  result_test={
      'test_loss':0,
      'test_acc':0,
  }

  if optimizer.upper()=='adam':
    optimizer=torch.optim.Adam(model.parameters(), lr=float(lr))
  else:
     optimizer=torch.optim.SGD(model.parameters(), lr=float(lr))



  for epoch in range(int(epochs)):
    train_loss, train_acc = train_step(model=model,dataloader=train_dataloader,loss_fn=loss_fn,optimizer=optimizer, device=device)
    print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f}"

      )
    result_train['epoch'].append(epoch)
    result_train['train_loss'].append(round(train_loss,5))
    result_train['train_acc'].append(round(train_acc,5))

  test_loss,test_acc=test_step(model=model,dataloader=test_dataloader,loss_fn=loss_fn, device=device)

  print(f"test_loss: {test_loss:.4f} test_acc: {test_acc:.4f}")
  result_test['test_loss']=round(test_loss,5)
  result_test['test_acc']=round(test_acc,5)
  return result_train,result_test