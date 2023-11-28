import requests
from enum import Enum
import sqlite3
import torch
from scripts import data_setup,model,engine,utils
import asyncio
import aiohttp
import requests
from client import Client
import mysql.connector
from mysql.connector import Error
import json
import os
from fedavg import fedAvgAlgorithm

client_url='http://127.0.0.1:5001'
server_url='http://127.0.0.1:5000/client'

# class Server_status(Enum):
#     IDLE=1,
#     TRAINING=2,
#     TRAINING_CLIENTS=3,
#     RUNNING=4



class Server:
    def __init__(self):
        self.status='IDLE'
        self.host='127.0.0.1'
        self.user='root'
        self.passwd='PASSWORD'
        self.database='federated_learning'
        self.init_database()
      
    def create_server_connection(self, host_name, user_name, user_password, database=''):
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
       
    def start_server(self,lr,epochs,batch_size,optim,rounds):
        self.learing_rate=lr
        self.epochs=epochs
        self.batch_size=batch_size
        self.optimizer=optim
        self.status='RUNNING'
        self.rounds=rounds
        self.clients=[]
        #print(f'Types {type(self.learing_rate)},{self.learing_rate}, {type(self.learing_rate)}, {type(self.epochs)}, {type(self.batch_size)}, {type(self.optimizer)}')
        connection = self.create_server_connection(self.host, self.user, self.passwd,self.database)
        cursor = connection.cursor()
        try:
            cursor.execute("DELETE FROM server")
            cursor.execute("INSERT INTO server (status,learning_rate,epochs,batch_size,optim,round) VALUES(%s,%s,%s,%s,%s,%s)", (self.status,float(self.learing_rate),int(self.epochs),int(self.batch_size),self.optimizer,self.rounds))
        except Error as err:
            print(f"Error: '{err}'")
        connection.commit()
        cursor.close()
        connection.close()

    def train(self,name='server_to_client',case='server'):
        train_dataloader,test_dataloader,n_classes,n_channels=data_setup.create_dataloaders_MNIST(32)
        model_g=model.Net(n_channels,n_classes)
        self.status='TRAINING'
        result_train,result_test=engine.train(
            model=model_g,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            epochs=4,
            optimizer='sgd',
            lr=0.002,
            case=case
                )
        print('Save model',result_train,result_test)
        utils.save_model(model=model_g,target_dir_path="data",model_name='model_net.pt',name=name)
        self.status='RUNNING'
        #print(f'Server status {self.status}')
        return [result_train,result_test]
    # def register(self,client_url:str,client_status):

    #     conn=sqlite3.connect('clients.db')
    #     cursor=conn.cursor()
    #     print('Client url',len(cursor.execute("SELECT client_url FROM clients").fetchall()))
    #     client_url_g=5000+len(cursor.execute("SELECT client_url FROM clients").fetchall())+1
    #     print('Generated client_url',client_url_g)
    #     #TODO:check if client exist in list
    #     self.clients.append({
    #         'client_url':client_url,
    #         'client_status':client_status
    #     })
    #     if cursor.execute('SELECT id FROM clients WHERE client_url=(?)',[client_url]).fetchall():
            
    #         client= cursor.execute('SELECT id,status FROM clients WHERE client_url=(?)',[client_url]).fetchall()
    #         if client:
    #             id,status=client[0]
    #             print(type(client),id,status) 
    #             if(client_status!=status):
    #                 print('update')
    #                 cursor.execute("UPDATE clients SET status = (?) WHERE id=(?)",[client_status,id])
    #     else:
    #         print('ADD client')
    #         cursor.execute("INSERT INTO clients (client_url,status) VALUES(?, ?)", (client_url,client_status))
        
    #     conn.commit()
    #     cursor.close()
    #     conn.close()
    #     print('Data received from client:', client_url,client_status)

    def register(self,client_name,lr,epochs,batch_size,optim):
        clients=[]
        client_status='Idle'
        connection = self.create_server_connection(self.host, self.user, self.passwd,self.database)
        cursor = connection.cursor(buffered=True)
        #print('Client url',len(cursor.execute("SELECT client_url FROM clients").fetchall()))
        try:
            cursor.execute("SELECT client_url FROM clients ORDER BY client_url DESC LIMIT 1")
            result=cursor.fetchall()[0]
            client_url=result[0]
            print(f'NEW CLIENT_URL {client_url}')
            url=int(client_url)+1
            client_url=str(url)
            print(f'NEW CLIENT_URL {client_url}')
        except Error as err:
            print(f"Error: '{err}'")
        #TODO:check if client exist in list
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
        #print('Data received from client:', client_url,client_status)
    async def delete_user(self,client_id):
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
       # print(f'RETURN CLIENTS {clients_list}') 
        self.clients=clients_list
        return clients_list
    def updateStatus(self,status:str):
        connection = self.create_server_connection(self.host, self.user, self.passwd,self.database)
        cursor = connection.cursor()
        try:
            cursor.execute("UPDATE server SET status=(%s)",[status])
        except Error as err:
            print(f"Error: '{err}'")
        cursor.close()
        connection.close()
    
    def selectServerParams(self):
        connection = self.create_server_connection(self.host, self.user, self.passwd,self.database)
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT * FROM server")
            server=cursor.fetchone()
            (id,status,lr,epoch,batch_size,optim,rounds)=server
        except Error as err:
            print(f"Error: '{err}'")
        cursor.close()
        connection.close()
        print('SELECTED server params',status,lr,epoch,batch_size,optim,rounds)
        return status,lr,epoch,batch_size,optim,rounds
    
    def selectStatus(self):
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
    
    
    async def start_training(self):
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
                result=[]
                for round_ in range(rounds):
                    result_train,result_test=self.train(name='server_for_client',case='server')
                    background_tasks = set()
                    for client in training_scenario[0]['clients']:
                        print(f'Client select to train {client}')
                        task = asyncio.create_task(self.training_client_request(client,round_))
                        background_tasks.add(task)
                    await asyncio.gather(*background_tasks)
                    print('Result from gather')
                    print('After client server training')
                    fedAvgAlgorithm()
                    result.append({
                            'round':round_,
                            'train':result_train,
                            'test':result_test,
                        })
                with open('result_server.json','a') as f:
                    result_json = json.dumps(result, indent=3)
                    f.write(f'{result_json }\n')

                    # data=pd.read_csv('Federated-Learning-Project/server/data/params.csv')
                    # df=pd.DataFrame(data=data)
                    # print(df.mean())
                    # print(df.describe())
                    
    
                    

                
                
                #print(f"There are {len(clients_training)}clients registered in the system")
                #print(f'Asyncio gather result {result}')
    
    async def manage_traing_porcess(self):

        training_scenario=[]
        clients_training=await self.select_client(training=True)
        status_server,lr_server,epochs_server,batch_size_server,optim_server,rounds= self.selectServerParams()
        # for i in range(len(clients_training)):
        #     clients_epochs.append({
        #         'client_name':clients_training[i]['client_name'],
        #         'epochs':clients_training[i]['epochs'],
        #     })
        # sorted_clients_training=sorted(clients_epochs,key=lambda d: d['epochs'],reverse=True)
        # if (epochs_server<sorted_clients_training[0]['epochs'] and epochs_server<sorted_clients_training[1]['epochs']):
        #     epochs_training=epochs_server
        #     print('Epochs for training',epochs_training)
        # elif (epochs_server<sorted_clients_training[0]['epochs'] and epochs_server>sorted_clients_training[1]['epochs']) or (epochs_server>sorted_clients_training[0]['epochs'] and epoch_server>sorted_clients_training[1]['epochs']):
        #     epochs_training=sorted_clients_training[1]['epochs']
        #     print('Epochs for training',epochs_training)

     
        training_scenario.append(
                {   
                    'rounds':rounds,
                    'server':{
                        'lr':lr_server,
                        'epochs':epochs_server,
                        'batch_size':batch_size_server,
                        'optim':optim_server
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
    

    async def training_client_request(self,client,round_):
        print(f'Session {client["url"]}')
        async with aiohttp.ClientSession() as session:
            async with session.post(f'http://127.0.0.1:5000/training/client_{client["url"]}') as response:
                if response.status != 200:
                    print('Error requesting training to client', client["url"])
                else:
                   
                    print('Client', client["url"], 'started training')
                    client=Client(client['url'],client['lr'],client['epochs'],client['batch_size'],client['optim'])
                    result_train,result_test=client.train(name=client.client_url)
                    print(f'{client.client_url,result_train,result_test }')
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



    
    def init_database(self):

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
                       round INTEGER 
                       )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (id INTEGER PRIMARY KEY AUTO_INCREMENT)
        """)


        cursor.close()
        connection.close()
