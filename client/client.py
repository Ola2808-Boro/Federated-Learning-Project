import requests
from enum import Enum
client_url='http://127.0.0.1:5002'
server_url='http://127.0.0.1:5000/client'

class Client_status(Enum):
    IDLE=1,
    TRAINING=2

class Client:
    def __init__(self,client_url:str,server_url:str,learing_rate:float,epochs:int,batch_size:int,optimizer:str):
        self.client_url=client_url
        self.server=server_url
        self.learing_rate=learing_rate
        self.epochs=epochs
        self.batch_size=batch_size
        self.optimizer=optimizer
        self.status=Client_status.TRAINING
        self.register()
    def register(self):
        print(type(self.client_url),client_url,type(self.status),self.status)
        response = requests.post(server_url, 
            data={
            'client_url': self.client_url,
            'client_status':self.status
            }, timeout=5)
        print('Response received from registration:', response)
     
    
client=Client(client_url,server_url,0.001,5,32,'adam')
