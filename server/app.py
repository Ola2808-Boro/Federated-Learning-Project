from flask import Flask,render_template, request, Response
from constants import header_title,server_status_idle,server_status_running,server_status_text,clients_status_text,clients_status_text_not_found
from server import Server
import asyncio
from server import Server_status
app = Flask(__name__)



server= Server(0.01,5,32,'adam')
data={}
result=[]
# clients=[]

@app.route("/",methods=['GET','POST'])
async def home_page():
    # clients=await server.select_client()
    # print('Request',clients)
    clients=[]
    if request.method=='GET':
        print('GET')
        clients.append(await server.select_client())
        print('GET clients',clients,await server.select_client())
    if request.method=='POST':
        print('POST')
        data.update({
            'lr':request.form.get('lr'),
            'epochs':request.form.get('epochs'),
            'batch_size': request.form.get('batch_size'),
            'optim':request.form.get('optim')
        })
        server.learing_rate=data['lr']
        server.batch_size=data['batch_size']
        server.epochs=data['epochs']
        server.optimizer=data['optim']
        server.status=Server_status.TRAINING
        #server= Server(learing_rate=data['lr'],epochs=data['epochs'],batch_size=data['batch_size'],optimizer=data['optim'])
        result_train,result_test=await server.train()
        result.append([result_train,result_test])
        server.status=Server_status.RUNNING
        print('POST clients',clients)
   
    
    return render_template(
            'index.html',
            header_title=header_title,
            server_status_text=server_status_text,
            clients_status_text=clients_status_text,
            clients_status_text_not_found=clients_status_text_not_found,
            data=data,
            result=result,
            clients=clients[0],
            num_clients=len(clients[0])
        )
    #     if result_train:
    #         return render_template(
    #         'server.html',
    #         header_title=header_title,
    #         server_status=server_status_running,
    #         server_status_running= 'END learning',
    #         server_status_text=server_status_text,
    #         clients_status_text=clients_status_text,
    #         clients_status_text_not_found=clients_status_text_not_found,
    #         data=data,
    #         result=[result_train,result_test]
    #         )
    #     else:
    #         return render_template(
    #         'server.html',
    #         header_title=header_title,
    #         server_status=server_status_running,
    #         server_status_running= 'Learning',
    #         server_status_text=server_status_text,
    #         clients_status_text=clients_status_text,
    #         clients_status_text_not_found=clients_status_text_not_found,
    #         data=data,
    #         )
        
    # else:

     

# @app.route("/server",methods=['POST'])
# def server_page():
#     print('Post')
#     return render_template(
#         'index.html',
#         header_title=header_title,
#         server_status=server_status_running,
#         server_status_text=server_status_text,
#         clients_status_text=clients_status_text,
#         clients_status_text_not_found=clients_status_text_not_found
#     )

    # return render_template(
    #     'index.html',
    #     header_title=header_title,
    #     server_status=server_status_running,
    #     server_status_text=server_status_text,
    #     clients_status_text=clients_status_text,
    #     clients_status_text_not_found=clients_status_text_not_found
    # )

@app.route('/client', methods=['POST'])
def register_client():
    # print('Request POST /client for client_url [', request.form['client_url'], ']')
    # server.register(client_url=request.form['client_url'],client_status=request.form['client_status'])
    return Response(status=201)

@app.route('/client/<string:CLIENT_NAME>/<string:LR>/<string:EPOCHS>/<string:BATCH_SIZE>/<string:OPTIM>', methods=['POST'])
def client(CLIENT_NAME,LR,EPOCHS,BATCH_SIZE,OPTIM):
    print('CLIENT add',CLIENT_NAME,LR,EPOCHS,BATCH_SIZE,OPTIM)
    server.register(CLIENT_NAME,LR,EPOCHS,BATCH_SIZE,OPTIM)
    return Response(status=200)


@app.route('/training', methods=['POST'])
def training():
    print('Training')
    asyncio.run(server.start_training())
    return Response(status=200)


@app.route('/delete_user/<int:CLIENT_ID>', methods=['POST'])
async def delete_user(CLIENT_ID):
    print('Delete user, clieny id:',CLIENT_ID)
    await server.delete_user(CLIENT_ID)
    return Response(status=200)
   


