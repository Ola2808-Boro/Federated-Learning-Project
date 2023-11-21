import requests
from enum import Enum
import sqlite3
import torch
from scripts import data_setup,model,engine,utils
import asyncio
import aiohttp
import requests
from client import Client

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
        self.init_database()
        self.host='http://127.0.0.1/'
        
       
    def start_server(self,lr,epochs,batch_size,optim,rounds):
        self.learing_rate=lr
        self.epochs=epochs
        self.batch_size=batch_size
        self.optimizer=optim
        self.status='RUNNING'
        self.rounds=rounds
        self.clients=[]
        #print(f'Types {type(self.learing_rate)},{self.learing_rate}, {type(self.learing_rate)}, {type(self.epochs)}, {type(self.batch_size)}, {type(self.optimizer)}')
        conn=sqlite3.connect('federated_learing.db')
        cursor=conn.cursor()
        cursor.execute("DELETE FROM server")
        cursor.execute("INSERT INTO server (status,learning_rate,epochs,batch_size,optim,round) VALUES(?,?,?,?,?,?)", (self.status,float(self.learing_rate),int(self.epochs),int(self.batch_size),self.optimizer,self.rounds))
        conn.commit()
        cursor.close()
        conn.close()

    def train(self):
        train_dataloader,test_dataloader,n_classes,n_channels=data_setup.create_dataloaders_MNIST(32)
        model_g=model.Net(n_channels,n_classes)
        self.status='TRAINING'
        result_train,result_test=engine.train(
            model=model_g,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            epochs=4,
            optimizer='sgd',
            lr=0.002
                )
        print('Save model')
        utils.save_model(model=model_g,target_dir_path="Federated-Learning-Project/server/data",model_name='model_net.pt')
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
        conn=sqlite3.connect('federated_learing.db')
        cursor=conn.cursor()
        #print('Client url',len(cursor.execute("SELECT client_url FROM clients").fetchall()))
        client_url=5000+len(cursor.execute("SELECT client_url FROM clients").fetchall())+1
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
        if cursor.execute('SELECT id FROM clients WHERE client_url=(?)',[client_url]).fetchall():
            
            client= cursor.execute('SELECT id,status FROM clients WHERE client_url=(?)',[client_url]).fetchall()
            if client:
                id,status=client[0]
                print(type(client),id,status) 
                if(client_status!=status):
                    print('update')
                    cursor.execute("UPDATE clients SET status = (?) WHERE id=(?)",[client_status,id])
        else:
            print('ADD client')
            cursor.execute("INSERT INTO clients (client_name,client_url,status,learning_rate,epochs,batch_size,optim) VALUES(?,?, ?,?, ?,?, ?)", (client_name,client_url,client_status,float(lr),int(epochs),int(batch_size),optim))
        
        conn.commit()
        cursor.close()
        conn.close()
        #print('Data received from client:', client_url,client_status)
    async def delete_user(self,client_id):
        conn=sqlite3.connect('federated_learing.db')
        cursor=conn.cursor()
        cursor.execute("DELETE FROM clients WHERE id=(?)",[client_id])
        conn.commit()
        cursor.close()
        conn.close()
    async def select_client(self,training=False):

        clients_list=[]
        # return self.clients
        conn=sqlite3.connect('federated_learing.db')
        cursor=conn.cursor()
        if training:
            print('TRAINING TRUE')
            clients=cursor.execute("SELECT * FROM clients WHERE status='Idle'").fetchall()
        else:
            clients=cursor.execute("SELECT * FROM clients").fetchall()
        print('SELECTED CLIENT',type(clients),clients)
        cursor.close()
        conn.close()
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
        conn=sqlite3.connect('federated_learing.db')
        cursor=conn.cursor()
        cursor.execute("UPDATE server SET status=(?)",[status])
        cursor.close()
        conn.close()
    
    def selectServerParams(self):
        conn=sqlite3.connect('federated_learing.db')
        cursor=conn.cursor()
        server=cursor.execute("SELECT * FROM server").fetchall()[0]
        (id,status,lr,epoch,batch_size,optim,rounds)=server
        cursor.close()
        conn.close()
        print('SELECTED server params',status,lr,epoch,batch_size,optim,rounds)
        return status,lr,epoch,batch_size,optim,rounds
    
    def selectStatus(self):
        conn=sqlite3.connect('federated_learing.db')
        cursor=conn.cursor()
        server=cursor.execute("SELECT status FROM server").fetchall()[0]
        status=server[0]
        cursor.close()
        conn.close()
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
                
                for round_ in range(rounds):
                    background_tasks = set()
                    for client in training_scenario[0]['clients']:
                        print(f'Client select to train {client}')
                        task = asyncio.create_task(self.training_client_request(client))
                        background_tasks.add(task)
                    await asyncio.gather(*background_tasks)
                    print('Result from gather')
                    self.train()
                    print('After server training')
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
    

    async def training_client_request(self,client):
        print(f'Session {client["url"]}')
        async with aiohttp.ClientSession() as session:
            async with session.post(f'http://127.0.0.1:5000/training/client_{client["url"]}') as response:
                if response.status != 200:
                    print('Error requesting training to client', client["url"])
                else:
                    print('Client', client["url"], 'started training')
                    client=Client(client['url'],client['lr'],client['epochs'],client['batch_size'],client['optim'])
                    result_train,result_test=client.train()
                    print(f'{client.client_url,result_train,result_test }')
                    with open('result.txt','a') as f:
                        print('Open file')
                        f.write(f'{client.client_url,result_train,result_test } \n')


    
    def init_database(self):
        conn=sqlite3.connect('federated_learing.db')
        cursor=conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clients(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       client_name TEXT  NOT NULL,
                       client_url TEXT UNIQUE NOT NULL,
                       status TEXT NOT NULL,
                       learning_rate FLOAT NOT NULL,
                       epochs INTEGER NOT NULL,
                       batch_size INTEGER NOT NULL,
                       optim TEXT  NOT NULL
                       )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS server(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       status TEXT NOT NULL,
                       learning_rate FLOAT NOT NULL,
                       epochs INTEGER NOT NULL,
                       batch_size INTEGER NOT NULL,
                       optim TEXT  NOT NULL,
                       round INTEGER NOT NULL
                       )
        """)

        # cursor.execute("""
        #     CREATE TABLE IF NOT EXISTS training(
        #                id INTEGER PRIMARY KEY AUTOINCREMENT,
        #                client_name TEXT NOT NULL,
        #                client_url TEXT NOT NULL,
        #                complete_epochs INTEGER NOT NULL,
        #                )
        # """)

        # cursor.execute("""
        #     CREATE TABLE IF NOT EXISTS training_results(
        #                id INTEGER PRIMARY KEY AUTOINCREMENT,
        #                FOREIGN KEY (id) REFERENCES words(ID),
        #                )
        # """)
        cursor.close()
        conn.close()
