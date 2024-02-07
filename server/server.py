import torch
from scripts import data_setup,models,engine,utils
from constants import datasets_name
import asyncio
import glob
import aiohttp
from client import Client
import mysql.connector
import shutil
from mysql.connector import Error
import json
import os
from datetime import datetime
from fedavg import fedAvgAlgorithm
from torch.utils.data import DataLoader

client_url='http://127.0.0.1:5001'
server_url='http://127.0.0.1:5000/client'


class Server:
    def __init__(self):

        """
            Description: Server initialization function.
        """
     
        self.status='IDLE'
        self.host='127.0.0.1'
        self.user='root'
        self.passwd='PASSWORD'
        self.database='federated_learning'
        self.init_database()
      
    def create_server_connection(self, host_name, user_name, user_password, database=''):

        """
            Description: Creating a connection to the database.

            Args:
            host_name - name of host
            user_name - name of user
            user_password - password
            database - name of database

            Returns: connection
        """
        print(database)
        connection = None
        if database=='':
            try:
                connection = mysql.connector.connect(
                    host=host_name,
                    user=user_name,
                    passwd=user_password
                )
                print("MySQL Database connection successful")
            except Error as err:
                print(f"Error: '{err}'")
        else:
            try:
                connection = mysql.connector.connect(
                    host=host_name,
                    user=user_name,
                    passwd=user_password,
                    database=database
                )
                print("MySQL Database connection successful")
            except Error as err:
                print(f"Error: '{err}'")
        return connection
       
    def start_server(self,lr,epochs,batch_size,optim,rounds,model,strategy,datasets):

        """
            Description: A function that gives a parameter to the server and adds it to the database.

            Args:
            lr - learing rate
            batch_size - batch-size
            optim - name of optimizer
            rounds - number of rounds
            model - model name
            strategy - name of strategy
            datasets - database(s) on which federated learning will be carried out
        """

        self.learing_rate=lr
        self.epochs=epochs
        self.batch_size=batch_size
        self.optimizer=optim
        self.status='RUNNING'
        self.rounds=rounds
        self.model=model
        self.strategy=strategy
        self.clients=[]
        self.datasets=datasets
        #print(f'Types {type(self.learing_rate)},{self.learing_rate}, {type(self.learing_rate)}, {type(self.epochs)}, {type(self.batch_size)}, {type(self.optimizer)}')
        connection = self.create_server_connection(self.host, self.user, self.passwd,self.database)
        cursor = connection.cursor()
        try:
            cursor.execute("DELETE FROM server")
            cursor.execute("INSERT INTO server (status,learning_rate,epochs,batch_size,optim,round,strategy,model,datasets) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)", (self.status,float(self.learing_rate),int(self.epochs),int(self.batch_size),self.optimizer,self.rounds,self.strategy,self.model,self.datasets))
            print('Add server')
        except Error as err:
            print(f"Error: '{err}'")
        connection.commit()
        cursor.close()
        connection.close()

    def test(self,model,training_scenario:dict,train_dataloader:DataLoader,test_dataloader:DataLoader,name='server_to_client',case='server',model_name='model_name'):
        
        """
            Description: Testing model for server.

            Args:
            training_scenario - plan of training
            train_dataloader - dataloader that is used to train the model
            test_dataloader - dataloader that is used to test the model
            name - file name to save
            case - a variable to correctly set the process depending on whether it is called on the server or the client
            model_name - model name


            Returns: results
        """
        
        
        #self.status='TRAINING'
        result_train,result_test=engine.train(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            epochs=training_scenario[0]['server']['epochs'],
            optimizer=training_scenario[0]['server']['optim'],
            lr=training_scenario[0]['server']['lr'],
            case=case,
            model_name=model_name
                )
        print('Save model',result_train,result_test)
        utils.save_model(model=model,target_dir_path="data",model_name='model_net.pt',name=name)
        self.status='RUNNING'
        #print(f'Server status {self.status}')
        return [result_train,result_test]

    def register(self,client_name,lr,epochs,batch_size,optim):

            
        """
              Description: Registering the client in the database.

              Args:
               client name - name of client
               lr - learing rate
               epochs - number of epochs
               batch_size - batch-size
               optim - name of optimizer
        """
        
        clients=[]
        client_status='Idle'
        connection = self.create_server_connection(self.host, self.user, self.passwd,self.database)
        cursor = connection.cursor(buffered=True)
        #print('Client url',len(cursor.execute("SELECT client_url FROM clients").fetchall()))
        try:
            cursor.execute("SELECT client_url FROM clients ORDER BY client_url DESC LIMIT 1")
            #print(cursor.fetchall(),len(cursor.fetchall()))
            if cursor.rowcount==0:
                client_url=5000
                print('Pusto jest')

            else:
                result=cursor.fetchall()[0]
                print(result)
                (client_url,)=result
            print(f'NEW CLIENT_URL {client_url}')
            url=int(client_url)+1
            client_url=str(url)
            print(f'NEW CLIENT_URL {client_url}')
        except Error as err:
            print(f"Error: '{err}'")

        clients.append({
            'client_name':client_name,
            'client_url':client_url,
            'client_status':client_status,
            'lr':lr,
            'epochs':epochs,
            'batch_size':batch_size,
            'optim':optim
        })
        if cursor.execute('SELECT id FROM clients WHERE client_url=(%s)',(client_url,)):
            try:
                cursor.execute('SELECT id,status FROM clients WHERE client_url=(%s)',(client_url,))
                client=cursor.fetchall()
            except Error as err:
                print(f"Error: '{err}'")
            if client:
                id,status=client[0]
                print(type(client),id,status) 
                if(client_status!=status):
                    print('update')
                    try:
                        cursor.execute("UPDATE clients SET status = (%s) WHERE id=(%s)",(client_status,id))
                    except Error as err:
                        print(f"Error: '{err}'")
        else:
            print('ADD client')
            try:
                cursor.execute("INSERT INTO clients (client_name,client_url,status,learning_rate,epochs,batch_size,optim) VALUES(%s,%s, %s,%s, %s,%s, %s)", (client_name,client_url,client_status,float(lr),int(epochs),int(batch_size),optim))
            except Error as err:
                print(f"Error: '{err}'")
        
        connection.commit()
        cursor.close()
        connection.close()
        print('Data received from client:', client_url,client_status)
    async def delete_user(self,client_id):

        """
              Description: Function that removes a client from the database.

              Args:
              client_id - id of the client to be deleted
        """
         
        connection = self.create_server_connection(self.host, self.user, self.passwd,self.database)
        cursor = connection.cursor()
        try:
            cursor.execute("DELETE FROM clients WHERE id=(%s)",(client_id,))
            print(f'DELET client with id {client_id}')
        except Error as err:
            print(f"Error: '{err}'")
        connection.commit()
        cursor.close()
        connection.close()
    async def select_client(self,training=False):

        
        """
              Description: A function that retrieves data of available clients from the database.

              Args:
              training - a variable checking whether the function is called during training

              Returns: client list
        """

        clients_list=[]
        # return self.clients
        connection = self.create_server_connection(self.host, self.user, self.passwd,self.database)
        cursor = connection.cursor()
        if training:
            print('TRAINING TRUE')
            try:
                cursor.execute("SELECT * FROM clients WHERE status='Idle'")
                clients=cursor.fetchall()
            except Error as err:
                print(f"Error: '{err}'")

        else:
            try:
                cursor.execute("SELECT * FROM clients")
                clients=cursor.fetchall()
            except Error as err:
                print(f"Error: '{err}'")
        print('SELECTED CLIENT',type(clients),clients)
        cursor.close()
        connection.close()
        if clients:
            #print(len(clients[0]),clients[0])
            for i in range(len(clients)):
                #print(f'Iteracja {i}')
                id,client_name,client_url,status,lr,epochs,batch_size,optim=clients[i]
                clients_list.append({
                    'id':id,
                    'client_name':client_name,
                    'client_url':client_url,
                    'client_status':status,
                    'lr':lr,
                    'epochs':epochs,
                    'batch_size':batch_size,
                    'optim':optim
                })
        print(f'RETURN CLIENTS {clients_list}') 
        self.clients=clients_list
        return clients_list
    def updateStatus(self,status:str):
        
        """
              Description: A function that updates server data in the database.
        """

        connection = self.create_server_connection(self.host, self.user, self.passwd,self.database)
        cursor = connection.cursor()
        try:
            cursor.execute("UPDATE server SET status=(%s)",[status])
        except Error as err:
            print(f"Error: '{err}'")
        cursor.close()
        connection.close()
    
    def addParam(self,status:str):
        pass
    def selectServerParams(self):

        """
              Description: A function that retrieves server data from the database.

              Returns: server params
        """
          
        connection = self.create_server_connection(self.host, self.user, self.passwd,self.database)
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT * FROM server")
            server=cursor.fetchone()
            (id,status,lr,epoch,batch_size,optim,rounds,model,strategy,datasets)=server
        except Error as err:
            print(f"Error: '{err}'")
        cursor.close()
        connection.close()
        print('SELECTED server params',status,lr,epoch,batch_size,optim,rounds,datasets)
        return status,lr,epoch,batch_size,optim,rounds,model,strategy,datasets
    
    def selectStatus(self):

        """
              Description: Server status checking function.

              Returns: server status
        """
        connection = self.create_server_connection(self.host, self.user, self.passwd,self.database)
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT status FROM server")
            server=cursor.fetchone()
            status=server[0]
        except Error as err:
            print(f"Error: '{err}'")
        cursor.close()
        connection.close()
        print('SELECTED server status',status)
        return status
    
    def organizing_files(self):

        """
              Description: A function that organizes files containing results.
        """
         
        print('Organizing files')
        now = datetime.now()
        current_time = now.strftime("%d-%m-%Y-%H-%M-%S")
        if os.path.isdir('C:/Users/olkab/Desktop/Federated Learning App/Federated-Learning-Project/server/old_data'):
            print('The dir old_data exists')
        else:
            os.makedirs('C:/Users/olkab/Desktop/Federated Learning App/Federated-Learning-Project/server/old_data/',exist_ok=True)
        files=glob.glob('C:/Users/olkab/Desktop/Federated Learning App/Federated-Learning-Project/server/data/*.pt')
        files_client=glob.glob('C:/Users/olkab/Desktop/Federated Learning App/Federated-Learning-Project/server/data/client/*.pt')
        print('Files to move',files,files_client)
        for file in files_client:       
            os.makedirs(f'C:/Users/olkab/Desktop/Federated Learning App/Federated-Learning-Project/server/old_data/{str(current_time)}/client',exist_ok=True)
            print(f"Move file from {file} to {file.split('/data')[0]+f'/old_data/client{str(current_time)}/client'}")
            os.rename(file,file.split('/data')[0]+f'/old_data/{str(current_time)}/client/{file.split("/client")[1]}')
        for file in files:       
            os.makedirs(f'C:/Users/olkab/Desktop/Federated Learning App/Federated-Learning-Project/server/old_data/{str(current_time)}',exist_ok=True)
            print(f"Move file from {file} to {file.split('/data')[0]+f'/old_data/{str(current_time)}'}")
            os.rename(file,file.split('/data')[0]+f'/old_data/{str(current_time)}/{file.split("/data")[1]}')
        if os.path.isfile('C:/Users/olkab/Desktop/Federated Learning App/Federated-Learning-Project/server/result_clients.json'):
            shutil.move('C:/Users/olkab/Desktop/Federated Learning App/Federated-Learning-Project/server/result_clients.json',f'C:/Users/olkab/Desktop/Federated Learning App/Federated-Learning-Project/server/old_data/{str(current_time)}/results_clients.json')
        if os.path.isfile('C:/Users/olkab/Desktop/Federated Learning App/Federated-Learning-Project/server/result_server.json'):
            print('Exist ')
            shutil.move('C:/Users/olkab/Desktop/Federated Learning App/Federated-Learning-Project/server/result_server.json',f'C:/Users/olkab/Desktop/Federated Learning App/Federated-Learning-Project/server/old_data/{str(current_time)}/result_server.json')
            
    
    async def start_training(self):
 
        
        """
              Description: A function that manages the entire training.
        """

        self.organizing_files()
        status=self.selectStatus()
        clients_training= await self.select_client(training=True)
        print('Clients,training',clients_training)
        print(f'Clients idle {clients_training}, clients training {self.clients}')
        if  status=='IDLE':
            print(f'Server is not ready for training yet, status {status}')
        elif  status=='TRAINING':
            print(f'Server is not ready for training yet, because is during training process, status {status}')
        elif status=='RUNNING':
            if len(clients_training)==0:
                print("There aren't any clients registered in the system")
            elif len(clients_training)<2:
                print("There aren't enough clients registered in the system")
            else:
                training_scenario=await self.manage_traing_porcess()
                rounds=training_scenario[0]['rounds']
                model_name=training_scenario[0]['model']
                datasets=training_scenario[0]['datasets']
                model=models.choose_model(model_name=model_name)
                print('Model in train server function',model_name,model)
                result=[]
                clients_num=len(training_scenario[0]['clients'])
                train_dataloaders,test_dataloaders,validate_dataloaders=data_setup.create_dataloaders(batch_size=training_scenario[0]['server']['batch_size'],clients_number=clients_num,datasets=datasets_name[datasets])
                print('Train_dataloaders',train_dataloaders[0])
                for round_ in range(rounds):
                    path=f'C:/Users/olkab/Desktop/Federated Learning App/Federated-Learning-Project/server/classification_reports/server/'
                    result_train,result_test=self.test(model=model,training_scenario=training_scenario,train_dataloader=train_dataloaders[0],test_dataloader=test_dataloaders[0],name='server_for_client',case='server',model_name=model_name)
                    #classification_data_test.to_csv(f"{path}/{model_name}_{round_}.csv")
                    background_tasks = set()
                    for idx in range(len(training_scenario[0]['clients'])):
                        print(f'Client select to train {training_scenario[0]["clients"][idx]}')
                        task = asyncio.create_task(self.training_client_request(training_scenario[0]["clients"][idx],round_,train_dataloaders[idx+1],test_dataloaders[idx+1],model,model_name))
                        background_tasks.add(task)
                    await asyncio.gather(*background_tasks)
                    print('Result from gather')
                    print('After client server training')
                    fedAvgAlgorithm(model)
                    result.append({
                            'round':round_,
                            'train':[],
                            'test':result_test,
                        })
                with open('result_server.json','a') as f:
                    result_json = json.dumps(result, indent=3)
                    f.write(f'{result_json }\n')
    
    def plot_charts(self,results,case):

        """
              Description: A function for plotting  graphs of the results obtained by the model.

              Args:
              results - the results obtained by the model
              case - a variable differentiating whether the function is performed on an object such as a server or a client


              Returns: Charts
        """
        images=utils.plot_charts(results=results,case=case)
        return images

    async def manage_traing_porcess(self):
        
        """
        Description: A function used to plan the federated learning process.

        Returns: training plan
        """
        training_scenario=[]
        clients_training=await self.select_client(training=True)
        status_server,lr_server,epochs_server,batch_size_server,optim_server,rounds,strategy,model,datasets= self.selectServerParams()
  
        training_scenario.append(
                {   
                    'rounds':rounds,
                    'model':model,
                    'strategy':strategy,
                    'datasets':datasets,
                    'server':{
                        'lr':lr_server,
                        'epochs':epochs_server,
                        'batch_size':batch_size_server,
                        'optim':optim_server,
                    },
                    'clients':[ {
                        'url':clients_training[i]['client_url'],
                        'lr':clients_training[i]['lr'],
                        'epochs':clients_training[i]['epochs'],
                        'batch_size':clients_training[i]['batch_size'],
                        'optim':clients_training[i]['optim'],
                        } 
                        for i in range(len(clients_training)) 
                        ]
                }
            )
        print(f'Training scenario',training_scenario,len(training_scenario))
        return training_scenario
    

    async def training_client_request(self,client,round_,train_dataloader,test_dataloader,model,model_name):
        
        """
        Description: Function to conduct training.

        Args:
        client - id of client
        round_ - id of round
        train_dataloader - dataloader that is used to train the model
        test_dataloader - dataloader that is used to test the model
        model - model to train
        model_name - model name

        """
        
        print(f'Session {client["url"]}')
        async with aiohttp.ClientSession() as session:
            async with session.post(f'http://127.0.0.1:5000/training/client_{client["url"]}') as response:
                if response.status != 200:
                    print('Error requesting training to client', client["url"])
                else:
                   
                    print('Client', client["url"], 'started training')
                    client=Client(client['url'],client['lr'],client['epochs'],client['batch_size'],client['optim'])
                    result_train,result_test=client.train(model=model,name=client.client_url,train_dataloader=train_dataloader,test_dataloader=test_dataloader,case='client',model_name=model_name)
                    print(f'{client.client_url,result_train,result_test }')
                    #path=f'C:/Users/olkab/Desktop/Federated Learning App/Federated-Learning-Project/server/classification_reports/clients/{client["url"]}'
                    #os.makedirs(f'{path}/test',exist_ok=True)
                    #os.makedirs(f'{path}/train',exist_ok=True)
                    #classification_report_test.to_csv(f"{path}/test/{model_name}_{client['lr']}_{client['epochs']}_{client['batch_size']}_{client['optim']}_{round_}.csv")
                    #classification_report_train.to_csv(f"{path}/train/{model_name}_{client['lr']}_{client['epochs']}_{client['batch_size']}_{client['optim']}_{round_}.csv")
                    # result.append({
                    #         'round':round_,
                    #         'train':result_train,
                    #         'test':result_test,
                    #     }) 
                    if os. path. exists('result_clients.json'):
                        with open('result_clients.json','r') as f:
                            results=json.load(f)
                            print('Jest juz key ',str(client.client_url), results,type(results))
                            if str(client.client_url) in results:

                                results[str(client.client_url)].append({
                                        'client_url':client.client_url,
                                        'round':round_,
                                        'train':result_train,
                                        'test':result_test,
                                    })
                                with open('result_clients.json','w') as f:
                                        result_json = json.dumps(results, indent=3)
                                        f.write(f'{result_json }\n')
                            else:
                                results.update({str(client.client_url):[]})
                                results[str(client.client_url)].append({
                                        'client_url':client.client_url,
                                        'round':round_,
                                        'train':result_train,
                                        'test':result_test,
                                    })
                                with open('result_clients.json','w') as f:
                                    result_json = json.dumps( results, indent=3)
                                    f.write(f'{result_json }\n')
                    else:
                        results_new={str(client.client_url):[]}
                        results_new[str(client.client_url)].append({
                                'client_url':client.client_url,
                                'round':round_,
                                'train':result_train,
                                'test':result_test,
                            })
    
                        with open('result_clients.json','w') as f:
                                result_json = json.dumps( results_new, indent=3)
                                f.write(f'{result_json }\n')
                                

    def predict(self,filename,model_name):

        """
        Description: The purpose of the function is to make prediction.

        Args:
        model_name - name of the model that should make predictions
        filename - name of the photo on which the prediction is to be made

        Returns: prediction
        """
        print('Filename',filename)
        input=data_setup.prepare_img_to_predict(device='cpu',filename=filename,model_name=model_name)
        model=models.choose_model(model_name)
        model.eval()
        output = model(input)
        y_pred_class=torch.softmax(output, dim=1).argmax(dim=1)
        print(f'Model predcited {y_pred_class}, {output}')
        return y_pred_class

    
    def init_database(self):


        """
            Description: Function that creates a database.
        """
        connection = self.create_server_connection(self.host, self.user, self.passwd)
        cursor = connection.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS federated_learning")
        cursor.close()
        connection.close()

        connection = self.create_server_connection(self.host, self.user, self.passwd,self.database)
        print('Connect after create database')
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clients(
                       id INTEGER PRIMARY KEY AUTO_INCREMENT,
                       client_name TEXT NOT NULL,
                       client_url VARCHAR(200) UNIQUE NOT NULL,
                       status TEXT NOT NULL,
                       learning_rate FLOAT NOT NULL,
                       epochs INTEGER NOT NULL,
                       batch_size INTEGER NOT NULL,
                       optim TEXT NOT NULL
                       )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS server(
                       id INTEGER PRIMARY KEY AUTO_INCREMENT,
                       status TEXT NOT NULL,
                       learning_rate FLOAT NOT NULL,
                       epochs INTEGER NOT NULL,
                       batch_size INTEGER NOT NULL ,
                       optim TEXT NOT NULL,
                       round INTEGER,
                       strategy TEXT,
                       model TEXT,
                       datasets TEXT NOT NULL
                       )
        """)


        cursor.close()
        connection.close()
