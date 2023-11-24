import torch
from pathlib import Path
import os
from torchsummary import summary
import re
import mysql.connector
from mysql.connector import Error
import csv
import pandas as pd

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
  # with open(file='Federated-Learning-Project/server/data/params.txt',mode='w',encoding="utf-8") as f:
  #           for parameter in model.parameters():
  #             f.write(f'{parameter.flatten()} \n')
  # format()
        
  torch.save(obj=model.state_dict(),f=model_save_path)


# def save_data(result=''):
#       with open('Federated-Learning-Project/server/data/params.csv','a') as f:
#         #print(result)
#         writer=csv.writer(f)   
#         writer.writerow(result)

# def save_to_database(result,model_name):
#     connection = mysql.connector.connect(
#                     host=host_name,
#                     user=user_name,
#                     passwd=user_password,
#                     database=database
#                 )
  
#     cursor = connection.cursor()
#     col=model_name
#     # cursor.execute("ALTER TABLE results ADD {} ".format())
#     weights=[]
#     columns='( '
#     for itm in range(len(result)):
#       print(itm, len(result),itm ==len(result)-1)
#       weights.append([float(result[itm])])
#       if itm== int(len(result)-1):
#         columns=columns+(f'ola_{itm} FLOAT )')
#       else:
#          columns=columns+(f'ola_{itm} FLOAT,')
#     #print('Columns',columns)
#     cursor.execute("ALTER TABLE results ADD COLUMN {} ".format(columns))
#     # try:
#     #   query="""INSERT INTO results (model1) VALUES (%s)"""
#     #   cursor.executemany(query,weights)
#     #   print('After add')
#     #   cursor.execute("SELECT * FROM results")
#     #   # print(cursor.fetchall())
#     # except Error as err:
#     #   print(f"Error: '{err}'")
#     connection.commit()
#     cursor.close()
#     connection.close()


# def format():
#     with open('Federated-Learning-Project/server/data/params.txt','r') as f:
#         lines=f.readlines()
#         lines_clear=''.join(lines).replace('[','').replace(']','').replace('[]','')
#         result_clear=re.findall(r'\d+',lines_clear)
#         save_data(result=result_clear)


# def format(model_name):
#     with open('Federated-Learning-Project/server/data/params.txt','r') as f:   
#       lines=f.readlines()
#       if not lines:
#         print('FILE IS EMPTY')
#       else:
#         for line in lines:
#           line=line.replace('[]',',').replace('[',',').replace(']',',')
#           with open('Federated-Learning-Project/server/data/params_clear.txt','a') as file:  
#              print(line)
#              file.write(line)
#     with open('Federated-Learning-Project/server/data/params_clear.txt','r') as file_db:  
#       content=file_db.read()
#       result=re.findall(r'\d+',content)
#       #print(result)
#       #save_to_database(result=result,model_name=model_name)
#       save_data(result=result)

#         for line in lines:
#             print(line)
#             result=re.findall(r'\d+',line)
#             with open('Federated-Learning-Project/server/Federated-Learning-Project/server/data/paramas_ola_result.txt','a') as file:  
#                 file.write(f'{result}')        
#     with open('Federated-Learning-Project/server/Federated-Learning-Project/server/data/paramas_ola_result.txt','r') as file_aa:  
#                 lines=file_aa.readlines()
#                 for line in lines: 
#                   print(line)
#                   line=line.replace('[]',',').replace('[',',').replace(']',',')
#                   with open('Federated-Learning-Project/server/Federated-Learning-Project/server/data/paramas_ola_result_clear.txt','a') as file_OLA:  
#                     file_OLA.write(f'{line}') 
# format()

# with open('Federated-Learning-Project/server/Federated-Learning-Project/server/data/paramas_ola_result_clear.txt','r') as file_OLA:  
#             file_content = file_OLA.read()
#             result=re.findall(r'\d+',file_content)
#             #print(result)
#             save_to_database(result,len(result))

