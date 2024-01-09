import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd

accuracy = Accuracy(task="binary", num_classes=2)


def train_step(model:nn.Module, dataloader:DataLoader,optimizer:torch.optim.Optimizer,loss_fn:nn.Module,device:torch.device,model_name:str):
  model.train()
  loss_avg=0
  acc_avg=0
  for batch,(x,y) in enumerate(dataloader):
    #print('Batch',batch,'x ahape',x.shape,'y')
    x,y=x.to(device),y.to(device)
    y_pred=model(x).squeeze()
    y = y.squeeze()
    #print('y_predicted',y_pred,y_pred.shape)
    #y_pred_class=torch.round(torch.sigmoid(y_pred)) #to use to measure accuracy
    y_pred_class=torch.round(torch.sigmoid(y_pred))
    #print('y_predicted_class',y_pred_class,y_pred_class.shape)
    loss=loss_fn(y_pred,y.float())
    loss_avg=loss_avg+loss.item()
    acc=accuracy(y_pred_class,y)
    acc_avg=acc_avg+acc
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  target_names=['no cancer','cancer']
  classification_report=classification_report(y, y_pred_class, target_names=target_names)
  classification_data=pd.DataFrame(classification_report).transpose()
  loss_avg=loss_avg/len(dataloader)
  acc_avg=acc_avg/len(dataloader)
  return loss_avg,acc_avg.item(),classification_data

def test_step(model:nn.Module, dataloader:DataLoader,loss_fn:nn.Module,device:torch.device,model_name:str):

  model.eval()
  with torch.inference_mode():
    loss_avg=0
    acc_avg=0
    for batch,(x,y) in enumerate(dataloader):
      x,y=x.to(device),y.to(device)
      y_pred=model(x).squeeze()
      y = y.squeeze()
      #print('y_predicted',y_pred,y_pred.shape)
      #y_pred_class=torch.round(torch.sigmoid(y_pred)) #to use to measure accuracy
      y_pred_class=torch.round(torch.sigmoid(y_pred))
      #print('y_predicted_class',y_pred_class,y_pred_class.shape)
      loss=loss_fn(y_pred,y.float())
      loss_avg=loss_avg+loss.item()
      acc=accuracy(y_pred_class,y)
      acc_avg=acc_avg+acc
    target_names=['no cancer','cancer']
    classification_report=classification_report(y, y_pred_class, target_names=target_names)
    classification_data=pd.DataFrame(classification_report).transpose()
    loss_avg=loss_avg/len(dataloader)
    acc_avg=acc_avg/len(dataloader)
    return loss_avg,acc_avg.item(),classification_data

def train(model:nn.Module, train_dataloader:DataLoader,test_dataloader:DataLoader,epochs:int,optimizer:str,lr:float,case:str,model_name:str):

  loss_fn= torch.nn.BCEWithLogitsLoss()
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

  params_1x = [param for name, param in model.named_parameters() if 'fc' not in str(name)]
  if optimizer.upper()=='adam':
    print('Optimizer ADAM')
    optimizer = torch.optim.Adam([{'params':params_1x}, {'params': model.fc.parameters(), 'lr': lr*10}], lr=lr, weight_decay=5e-4)
  else:
     optimizer=torch.optim.SGD(model.parameters(), lr=float(lr))

  for epoch in range(int(epochs)):
    train_loss, train_acc,classification_data_train = train_step(model=model,dataloader=train_dataloader,loss_fn=loss_fn,optimizer=optimizer, device=device,model_name=model_name)
    print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f}"

      )
    result_train['epoch'].append(epoch)
    result_train['train_loss'].append(round(train_loss,5))
    result_train['train_acc'].append(round(train_acc,5))

  test_loss,test_acc,classification_data_test=test_step(model=model,dataloader=test_dataloader,loss_fn=loss_fn, device=device,model_name=model_name)

  print(f"test_loss: {test_loss:.4f} test_acc: {test_acc:.4f}")
  result_test['test_loss']=round(test_loss,5)
  result_test['test_acc']=round(test_acc,5)
  return result_train,result_test,classification_data_train,classification_data_test