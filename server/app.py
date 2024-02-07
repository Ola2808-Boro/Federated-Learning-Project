from flask import Flask,render_template, request, Response
from constants import header_title,server_status_text,clients_status_text,clients_status_text_not_found
from server import Server
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Federated-Learning-Project\server\images'

server=Server()
data={}
result=[]


@app.route("/",methods=['GET','POST'])
async def home_page():

    """
      Description: View the main page of the application
    """

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
            datasets=request.form.get('datasets')
            print('Parametry z form',model,strategy)
            server.start_server(lr,epochs,batch_size,optim,rounds,model,strategy,datasets)
            server.updateStatus('RUNNING')
            data.update({
                'lr':lr,
                'epochs':epochs,
                'batch_size': batch_size,
                'optim':optim,
                'round':rounds,
                'model':model,
                'strategy':strategy,
                'datasets':datasets
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

    """
      Description: Displaying the results page for customers
    """
     
    print('Clients_result')
    with open('result_clients.json','r') as f:
        results=json.load(f)
        images,summary_images=server.plot_charts(results,'clients')
    return render_template(
            'results_clients.html',
            header_title=header_title,
            clients_status_text=clients_status_text,
            data=data,
            results=results,
            images=images,
             summary_images=summary_images
    )


@app.route('/server_result', methods=['GET','POST'])
async def server_result():
    """
      Description: Displaying the results page for server
    """
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
            images=images,
        )


@app.route('/client', methods=['POST'])
def register_client():
    """
      Description: Endpoint for adding clients
    """
    return Response(status=201)

@app.route('/client/<string:CLIENT_NAME>/<string:LR>/<string:EPOCHS>/<string:BATCH_SIZE>/<string:OPTIM>', methods=['POST'])
def client(CLIENT_NAME,LR,EPOCHS,BATCH_SIZE,OPTIM):
    """
      Description: Endpoint for adding clients
    """
    print('CLIENT add',CLIENT_NAME,LR,EPOCHS,BATCH_SIZE,OPTIM)
    server.register(CLIENT_NAME,LR,EPOCHS,BATCH_SIZE,OPTIM)
    return Response(status=200)



@app.route('/training', methods=['POST'])
async def training():
    print('Training')
    """
      Description: Endpoint for starting training
    """
    await server.start_training()
    return Response(status=200)


@app.route('/training/client_<string:URL>', methods=['POST'])
async def training_client(URL):
    """
      Description: Endpoint for training clients
    """
    print('Training')
    # await server.select_client()
    print('URL',URL)
    return Response(status=200)

@app.route('/delete_user/<int:CLIENT_ID>', methods=['POST'])
async def delete_user(CLIENT_ID):
    """
      Description: Endpoint for deleting clients
    """
    print('Delete user, clieny id:',CLIENT_ID)
    await server.delete_user(CLIENT_ID)
    return Response(status=200)
   

@app.route('/upload_image', methods=['GET','POST'])
async def upload_imaget():
    """
      Description: Endpoint for uploading image
    """
    return render_template(
            'predict.html',
            header_title=header_title,
            clients_status_text='Attach a photo and get predictions'
            
    )

@app.route('/predict', methods=['GET','POST'])
async def predict():
    """
      Description: Displaying the predict page for server
    """
    prediction_arr=[]
    if request.method=='POST':
        file=request.files['upload_img']
        model_name=request.form.get('model_pred')
        print('Request',request.form.get('model_pred'))
        file.save(file.filename)
        print('Done')
        prediction=server.predict(filename=file.filename,model_name=model_name)
        prediction_arr.append(prediction)
        if prediction==1:
            prediction_text='cancer'
        elif prediction==0:
            prediction_text='no cancer'
    return render_template(
            'predict.html',
            header_title=header_title,
            clients_status_text= 'Attach a photo and get predictions' if len(prediction_arr)==0 else f'Predicted {prediction_arr[0].item()}',
            prediction=prediction_arr

            
    )
