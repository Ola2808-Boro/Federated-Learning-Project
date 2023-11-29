from flask import Flask,render_template, request, Response
from constants import header_title,server_status_idle,server_status_running,server_status_text,clients_status_text,clients_status_text_not_found
from server import Server
import json
app = Flask(__name__)


server=Server()
data={}
result=[]
# clients=[]

@app.route("/",methods=['GET','POST'])
async def home_page():
    # clients= await server.select_client()
    # print('Request',clients)
    
    clients=[]
    if request.method=='GET':
        print('GET')
        clients.append(await server.select_client())
        print('GET clients',clients,await server.select_client())
    if request.method=='POST':
        print('POST')
        if server.status=='IDLE':
            lr=request.form.get('lr')
            epochs=request.form.get('epochs')
            batch_size=request.form.get('batch_size')
            optim=request.form.get('optim')
            rounds=request.form.get('round')
            model=request.form.get('model')
            strategy=request.form.get('aggregation')
            print('Parametry z form',model,strategy)
            server.start_server(lr,epochs,batch_size,optim,rounds,model,strategy)
            server.updateStatus('RUNNING')
            data.update({
                'lr':lr,
                'epochs':epochs,
                'batch_size': batch_size,
                'optim':optim,
                'round':rounds,
                'model':model,
                'strategy':strategy
            })

   
    
    return render_template(
            'index.html',
            header_title=header_title,
            server_status_text=server_status_text,
            clients_status_text=clients_status_text,
            clients_status_text_not_found=clients_status_text_not_found,
            data=data,
            result=result,
            clients=clients[0] if len(clients)!=0 else [],
            num_clients=len(clients[0]) if len(clients)!=0 else 0
        )


@app.route('/clients_result', methods=['GET','POST'])
async def clients_result():
    print('Clients_result')
    with open('result_clients.json','r') as f:
        results=json.load(f)
        images=server.plot_charts(results,'clients')
    return render_template(
            'results_clients.html',
            header_title=header_title,
            clients_status_text=clients_status_text,
            data=data,
            results=results,
            images=images
    )


@app.route('/server_result', methods=['GET','POST'])
async def server_result():
    print('Server_result')
    with open('result_server.json','r') as f:
        results=json.load(f)
        images=server.plot_charts(results,'server')
        return render_template(
            'results_server.html',
            header_title=header_title,
            server_status_text='Server',
            data=data,
            results=results,
            images=images
        )
    # await server.select_client()
    print('URL',URL)
    # return Response(status=200)

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

# @app.route('/strategy', methods=['POST'])
# async def strategy():
#     print('Choose strategy')
#     # await server.select_client()
#     await server.start_training()
#     return Response(status=200)

@app.route('/training', methods=['POST'])
async def training():
    print('Training')
    # await server.select_client()
    await server.start_training()
    return Response(status=200)


@app.route('/training/client_<string:URL>', methods=['POST'])
async def training_client(URL):
    print('Training')
    # await server.select_client()
    print('URL',URL)
    return Response(status=200)

@app.route('/delete_user/<int:CLIENT_ID>', methods=['POST'])
async def delete_user(CLIENT_ID):
    print('Delete user, clieny id:',CLIENT_ID)
    await server.delete_user(CLIENT_ID)
    return Response(status=200)
   


