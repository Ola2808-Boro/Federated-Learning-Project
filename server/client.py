import requests
from enum import Enum
from scripts import data_setup,model,engine,utils

client_url='http://127.0.0.1:5002'
server_url='http://127.0.0.1:5000/client'

class Client_status(Enum):
    IDLE=1,
    TRAINING=2

class Client:
    def __init__(self,client_url:str,learing_rate:float,epochs:int,batch_size:int,optimizer:str):
        self.client_url=client_url
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
     
    def train(self,name:str,case='client'):
        train_dataloader,test_dataloader,n_classes,n_channels=data_setup.create_dataloaders_MNIST(self.batch_size)
        model_g=model.Net(n_channels,n_classes)
        self.status='TRAINING'
        print(f'Server status {self.status}')
        result_train,result_test=engine.train(
            model=model_g,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            epochs=self.epochs,
            optimizer=self.optimizer,
            lr=self.learing_rate,
            case=case
                )
        utils.save_model(model=model_g,target_dir_path="data/client",model_name='model_net.pt',name=name)
        # self.status='RUNNING'
        #print(f'Server status {self.status}')
        return [result_train,result_test]
    
# client=Client(client_url,server_url,0.001,5,32,'adam')
