import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

accuracy = Accuracy(task="multiclass", num_classes=2)


def train_step(model:nn.Module, dataloader:DataLoader,optimizer:torch.optim.Optimizer,loss_fn:nn.Module,device:torch.device,model_name:str):

  """
  Description: The purpose of the function is to train the model.

  Args:
  model - model to train
  dataloader - dataloader that is used to train the model
  optimizer - optimizer involved in updating the model weights
  loss_fn - loss function needed to calculate the error
  device - device on which the model will be trained (CPU, GPU)

  Returns: metrics
  """

  model.train()
  loss_avg=0
  acc_avg=0
  for batch,(x,y) in enumerate(dataloader):
    x,y=x.to(device),y.to(device)
    y_pred=model(x).squeeze()
    y = y.squeeze()
    y_pred_class=torch.softmax(y_pred, dim=1).argmax(dim=1)
    loss=loss_fn(y_pred,y)
    loss_avg=loss_avg+loss.item()
    acc=accuracy(y_pred_class,y)
    acc_avg=acc_avg+acc
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  target_names=['no cancer','cancer']
  # classification_report_data=classification_report(y, y_pred_class, target_names=target_names)
  # classification_data=pd.DataFrame(classification_report_data).transpose()
  f1 = f1_score(y.cpu(), y_pred_class, average='weighted')
  conf_matrix = confusion_matrix(y.cpu(), y_pred_class)
  # Extract TP, TN, FP, FN
  TP = conf_matrix[1, 1]  # True Positives
  TN = conf_matrix[0, 0]  # True Negatives
  FP = conf_matrix[0, 1]  # False Positives
  FN = conf_matrix[1, 0]  # False Negatives

  # Calculate Sensitivity and Specificity
  sensitivity = TP / (TP + FN)
  specificity = TN / (TN + FP)

  test_epoch_f1 = f1_score(y.cpu(), y_pred_class, average='weighted')
  loss_avg=loss_avg/len(dataloader)
  acc_avg=acc_avg/len(dataloader)
  return loss_avg,acc_avg.item(), sensitivity, specificity,f1
  # return loss_avg,acc_avg.item(),classification_data

def test_step(model:nn.Module, dataloader:DataLoader,loss_fn:nn.Module,device:torch.device,model_name:str):
  """
  Description: The purpose of the function is to test the model.

  Args:
  model - model to test
  dataloader - dataloader that is used to test the model
  loss_fn - loss function needed to calculate the error
  device - device on which the model will be trained (CPU, GPU)

  Returns: metrics
  """
   
  model.eval()
  with torch.inference_mode():
    loss_avg=0
    acc_avg=0
    for batch,(x,y) in enumerate(dataloader):
      x,y=x.to(device),y.to(device)
      y_pred=model(x).squeeze()
      y = y.squeeze()
      y_pred_class=torch.softmax(y_pred, dim=1).argmax(dim=1)
      loss=loss_fn(y_pred,y)
      loss_avg=loss_avg+loss.item()
      acc=accuracy(y_pred_class,y)
      acc_avg=acc_avg+acc
    target_names=['no cancer','cancer']
    f1 = f1_score(y.cpu(), y_pred_class, average='weighted')
    # classification_report_data=classification_report(y, y_pred_class, target_names=target_names)
    # classification_data=pd.DataFrame(classification_report_data).transpose()
    conf_matrix = confusion_matrix(y.cpu(), y_pred_class)
    # Extract TP, TN, FP, FN
    TP = conf_matrix[1, 1]  # True Positives
    TN = conf_matrix[0, 0]  # True Negatives
    FP = conf_matrix[0, 1]  # False Positives
    FN = conf_matrix[1, 0]  # False Negatives

    # Calculate Sensitivity and Specificity
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    loss_avg=loss_avg/len(dataloader)
    acc_avg=acc_avg/len(dataloader)
    loss_avg=loss_avg/len(dataloader)
    acc_avg=acc_avg/len(dataloader)
    # return loss_avg,acc_avg.item(),classification_data
    return loss_avg,acc_avg.item(), sensitivity, specificity,f1

def train(model:nn.Module, train_dataloader:DataLoader,test_dataloader:DataLoader,epochs:int,optimizer:str,lr:float,case:str,model_name:str):

  """
  Description: The purpose of the function is to test the model.

  Args:
  model - model to test
  train_dataloader - dataloader that is used to test the model
  test_dataloader - dataloader that is used to test the model
  epochs- the number of epochs for which the model is to be trained
  optimizer - optimizer involved in updating the model weights
  lr - learining rate involved in updating the model weights
  case - a variable differentiating whether the function is performed on an object such as a server or a client, decides whether train_step and test_step or only test_step should be performed

  Returns: metrics
  """
    
  loss_fn= torch.nn.CrossEntropyLoss()
  device='cuda' if torch.cuda.is_available() else 'cpu'
  result_train={
      'epoch':[],
      'train_loss':[],
      'train_acc':[],
      'train_sensitivity':[],
      'train_specificity':[],
      'train_f1_score':[],

  }

  result_test={
      'test_loss':0,
      'test_acc':0,
      'test_sensitivity':0, 
      'test_specificity':0,
      'test_f1_score':0,
  }

  params_1x = [param for name, param in model.named_parameters() if 'fc' not in str(name)]
  if optimizer.upper()=='adam':
    print('Optimizer ADAM')
    optimizer = torch.optim.Adam([{'params':params_1x}, {'params': model.fc.parameters(), 'lr': lr*10}], lr=lr, weight_decay=5e-4)
  else:
     optimizer=torch.optim.SGD(model.parameters(), lr=float(lr))
  if case!='server':
    for epoch in range(int(epochs)):
      train_loss, train_acc, train_sensitivity, train_specificity,train_f1_score = train_step(model=model,dataloader=train_dataloader,loss_fn=loss_fn,optimizer=optimizer, device=device,model_name=model_name)
      print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f}"

        )
      result_train['epoch'].append(epoch)
      result_train['train_loss'].append(round(train_loss,5))
      result_train['train_acc'].append(round(train_acc,5))
      result_train['train_specificity'].append(round(train_specificity,5))
      result_train['train_sensitivity'].append(round(train_sensitivity,5))
      result_train['train_f1_score'].append(round(train_f1_score,5))

  test_loss,test_acc,test_sensitivity, test_specificity, test_f1_score=test_step(model=model,dataloader=test_dataloader,loss_fn=loss_fn, device=device,model_name=model_name)

  print(f"test_loss: {test_loss:.4f} test_acc: {test_acc:.4f}")
  result_test['test_loss']=round(test_loss,5)
  result_test['test_acc']=round(test_acc,5)
  result_test['test_specificity']=round(test_specificity,5)
  result_test['test_sensitivity']=round(test_sensitivity,5)
  result_test['test_f1_score']=round(test_f1_score,5)
  return result_train,result_test