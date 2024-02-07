import torch
from pathlib import Path
import os
from torchsummary import summary
from matplotlib.figure import Figure
import base64
from io import BytesIO



def save_model(model:torch.nn.Module,target_dir_path:str,model_name:str,name:str):

  """
        Description: The function is used to save model weights to a file.

        Args:
        model - model
        target_dir_path - name of the folder where the file will be saved
        model_name - name of the model
        name - file name
  """

  if Path(target_dir_path).is_dir():
    print(f'Directory {target_dir_path} already exists')
  else:
    os.makedirs(target_dir_path,exist_ok=True)
    print(f'Creating {target_dir_path} directory')

  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path +'/'+name+'_'+model_name
  print('File to save model',model_save_path)
  torch.save(obj=model.state_dict(),f=model_save_path)



def plot_summary_charts(results,case,color,key=''):
        
        
        """
              Description: A function for plotting summary graphs of the results obtained by the model.

              Args:
              results - the results obtained by the model
              case - a variable differentiating whether the function is performed on an object such as a server or a client
              color - chart color
              key - id of client

              Returns: Charts
        """

        images=[]
        loss=[]
        acc=[]
        sensitivity=[]
        specificity=[]
        f1_score=[]
        if case=='server':
          for index in range(len(results)):
            loss.append(results[index]['test']['test_loss'])
            acc.append(results[index]['test']['test_acc'])
            specificity.append(results[index]['test']['test_specificity'])
            sensitivity.append(results[index]['test']['test_sensitivity'])
            f1_score.append(results[index]['test']['test_f1_score'])
        elif case=='clients':
              for index in range(len(results[key])):
                loss.append(results[key][index]['test']['test_loss'])
                acc.append(results[key][index]['test']['test_acc'])
                specificity.append(results[key][index]['test']['test_specificity'])
                sensitivity.append(results[key][index]['test']['test_sensitivity'])
                f1_score.append(results[key][index]['test']['test_f1_score'])
              print(f'Summary data; Loss {len(loss)} acc {len(acc)} specificity {len(specificity)} sensitivity {len(sensitivity)} f1-score {len(f1_score)}')

        data_server=[loss,acc,sensitivity,specificity,f1_score]
        print(data_server)
        titles=['Loss','Accuracy','Sensitivity','Specificity','F1-score']
        for index in range(len(data_server)):
            fig = Figure()
            ax =fig.subplots()
            ax.plot(data_server[index],color,label=titles[index])
            ax.set_title(f'{case.capitalize()} {key} test - {titles[index]} ')
            ax.set_xlabel('Epochs')
            ax.set_ylabel(f'Test values - {titles[index]}')
            ax.legend(loc='upper right')
            buf = BytesIO()
            fig.savefig(buf, format="png")
            data = base64.b64encode(buf.getbuffer()).decode("ascii")
            images.append(f"data:image/png;base64,{data}")
        print(f'For {key}: len images {len(images)}')
        return images


def plot_chart(x,y,label,key,round,color):
    
    """
      Description: A function for plotting  graphs of the results obtained by the model.

      Args:
              x - x daat
              y - y data
              label - plotting values
              key - id of client
              round - round
              color - chart color
              

      Returns: Chart
    """
         
    fig = Figure()
    ax =fig.subplots()
    ax.plot(x,y,color,label=label)     
    ax.set_title(f'Client {key} training - round {round+1}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Train values')
    ax.legend(loc='upper right')
    buf = BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"data:image/png;base64,{data}"
    
def plot_charts(results,case):
    

         
    """
      Description: A function for plotting all charts, both for individual metrics and summary ones.

      Args:
              results - the results obtained by the model
              case - a variable differentiating whether the function is performed on an object such as a server or a client
      Returns: Charts
    """

    if case=='server':
            summary_images=plot_summary_charts(results=results,case=case,color='b')
            return summary_images
    
    elif case=='clients':
        summary_images={}
        images={}
        client_images=[]
        client_summary_images=[]
        for key in results.keys():
            for result in results[key]:
                chart_loss=plot_chart(result['train']['epoch'],result['train']['train_loss'],'loss',key,result["round"],'b')
                chart_acc=plot_chart(result['train']['epoch'],result['train']['train_acc'],'acc',key,result["round"],'g')
                chart_specificity=plot_chart(result['train']['epoch'],result['train']['train_specificity'],'specificity',key,result["round"],'c')
                chart_sensitivity=plot_chart(result['train']['epoch'],result['train']['train_sensitivity'],'sensitivity',key,result["round"],'m')
                chart_f1_score=plot_chart(result['train']['epoch'],result['train']['train_f1_score'],'f1_score',key,result["round"],'y')   
                client_images.append([chart_loss,chart_acc,chart_specificity,chart_sensitivity,chart_f1_score])

            client_summary_images.append(plot_summary_charts(results=results,color='b',case=case,key=key))
           
            images.update({key:client_images})
            summary_images.update({key:client_summary_images})
            client_images=[]
            client_summary_images=[]
        for key in summary_images.keys():
              print('Sum',key,len(summary_images[key]))
        for key in images.keys():
              print('Img',key,len(images[key]))
        return images,summary_images
