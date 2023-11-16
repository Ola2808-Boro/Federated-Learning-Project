import requests
from enum import Enum
import sqlite3
import torch
from scripts import data_setup,model,engine,utils
import asyncio
client_url='http://127.0.0.1:5001'
server_url='http://127.0.0.1:5000/client'

class Server_status(Enum):
    IDLE=1,
    TRAINING=2,
    TRAINING_CLIENTS=3,
    RUNNING=4

class Server:
    def __init__(self,learing_rate:float,epochs:int,batch_size:int,optimizer:str):
        self.learing_rate=learing_rate
        self.epochs=epochs
        self.batch_size=batch_size
        self.optimizer=optimizer
        self.status=Server_status.IDLE
        self.clients=[]
        self.init_database()
        self.host='http://127.0.0.1/'
        
       
    
    async def train(self):
        train_dataloader,test_dataloader,n_classes,n_channels=data_setup.create_dataloaders_MNIST(self.batch_size)
        model_g=model.Net(n_channels,n_classes)
        self.status=Server_status.TRAINING
        print(f'Server status {self.status}')
        result_train,result_test=engine.train(
            model=model_g,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            epochs=self.epochs,
            optimizer=self.optimizer,
            lr=self.learing_rate
                )
        utils.save_model(model=model_g,target_dir_path="Federated-Learning-Project/server/data",model_name='model_net.pt')
        self.status=Server_status.RUNNING
        print(f'Server status {self.status}')
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
        client_status='Idle'
        conn=sqlite3.connect('clients.db')
        cursor=conn.cursor()
        print('Client url',len(cursor.execute("SELECT client_url FROM clients").fetchall()))
        client_url=5000+len(cursor.execute("SELECT client_url FROM clients").fetchall())+1
        #TODO:check if client exist in list
        self.clients.append({
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
        print('Data received from client:', client_url,client_status)
    async def delete_user(self,client_id):
        conn=sqlite3.connect('clients.db')
        cursor=conn.cursor()
        cursor.execute("DELETE FROM clients WHERE id=(?)",[client_id])
        conn.commit()
        cursor.close()
        conn.close()
    async def select_client(self,training=False):

        clients_list=[]
        # return self.clients
        conn=sqlite3.connect('clients.db')
        cursor=conn.cursor()
        if not training:
            clients=cursor.execute("SELECT * FROM clients").fetchall()
        else:
            clients=cursor.execute("SELECT * FROM clients WHERE status=='IDLE'").fetchall()
        print(type(clients),clients)
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
        print(f'RETURN CLIENTS {clients_list}') 
        self.clients=clients_list
        return clients_list
    
    async def start_training(self):
        print(self.clients)
        clients_training=await self.select_client(training=True)
        print(f'Clients idle {clients_training}, clients training {self.clients}')
        if self.status==Server_status.IDLE:
            print(f'Server is not ready for training yet, status {self.status}')
        elif self.status==Server_status.TRAINING:
            print(f'Server is not ready for training yet, because is during training process, status {self.status}')
        elif self.status==Server_status.RUNNING:
            if len(clients_training)==0:
                print("There aren't any clients registered in the system")
            else:
                background_tasks = set()
                for client in clients_training:
                    task = asyncio.create_task()
                    background_tasks.add(task)
                result=asyncio.gather(*background_tasks)
                print(f"There are {len(clients_training)}clients registered in the system")
                print(f'Asyncio gather result {result}')
    def init_database(self):
        conn=sqlite3.connect('clients.db')
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
        cursor.close()
        conn.close()