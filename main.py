import logging
import os
import csv
from datetime import datetime, timedelta
from typing import List

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer 
from fastapi.responses import Response, JSONResponse, FileResponse
from jose import JWTError, jwt
from sqlalchemy import and_
from sqlalchemy.orm import Session
from sqlalchemy import func

import uuid

import docker
import models
import shutil
import hashing
import schemas
import requests
from time import time
from docker.errors import BuildError, APIError, ContainerError, ImageNotFound, NotFound
from utils import get_free_port, get_host_ip, file2zip
from database import SessionLocal, db_engine
from get_models import get_model_1, get_model_2, get_model_3

SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

origins = [
    "*"
]

    # "http://localhost.tiangolo.com",
    # "https://localhost.tiangolo.com",
    # "http://localhost",
    # "http://localhost:8080",
    # "http://localhost:3000",
    
app = FastAPI()
# TODO: Change if deployed with docker
docker_cli = docker.DockerClient(base_url='unix:///var/run/docker.sock')

if not os.path.exists(os.getcwd() + '/downloads'):
    os.makedirs(os.getcwd() + '/downloads')

if not os.path.exists(os.getcwd() + '/trainings'):
    os.makedirs(os.getcwd() + '/trainings')

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_current_user(data: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    return verify_token(data, credentials_exception)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token:str,credentials_exception):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = schemas.TokenData(email=email)
        return payload
    except JWTError:
        raise credentials_exception


models.Base.metadata.create_all(db_engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post('/user')
def create_user(request: schemas.User,db: Session = Depends(get_db)):
    new_user = models.User(name=request.name,email=request.email, password=hashing.Hash.bcrypt(request.password))
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    os.mkdir(os.getcwd() + f"/downloads/{new_user.id}_stored_files/")
    return new_user

@app.get('/user/{id}', response_model=schemas.ShowUser)
def get_user(id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"User with the id {id} is not available")
    return user

@app.post('/login')
#def login(request: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
def login(request: schemas.Login, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.name == request.username).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Invalid Credentials")

    if not hashing.Hash.verify(user.password, request.password):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Incorrect password")

    access_token = create_access_token(data={"sub": user.email, "id": user.id, "name": user.name})
    return {"access_token": access_token, "token_type": "Bearer", "user": {"id": user.id, "name": user.name }}

@app.get('/users')
def getData(db: Session = Depends(get_db), current_user: schemas.ShowUser = Depends(get_current_user)):
    users = db.query(models.User).filter(models.User.id != current_user['id']).all()
    return users

import os
from io import BytesIO

import pandas as pd

# TODO: Comment when deploying
@app.on_event("startup")
async def startup_event():
    logger = logging.getLogger("uvicorn.access")
    logger.setLevel(logging.DEBUG)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(),db: Session = Depends(get_db), current_user: schemas.ShowUser = Depends(get_current_user)):
    fs = await file.read()

    new_file = models.FileCollection(filename=file.filename, filesize= f'{len(fs)/1000} kb', user_id=current_user["id"])
    db.add(new_file)
    # file_name = "id_" + str(new_file.id) + "_" + file.filename
    # new_file.filename = file_name
    db.commit()
    db.refresh(new_file)

    file_name = "id_" + str(new_file.id) + "_" + file.filename
    new_file.filename = file_name
    db.commit()
    db.refresh(new_file)


    # file_location = os.path.join(f"../{current_user['id']}_stored_files/", file_name)
    file_location = os.path.join(os.getcwd() + f"/downloads/{current_user['id']}_stored_files/", file_name)
    with open(file_location, "wb+") as file_object:
        file_object.write(fs)

    return new_file

@app.get("/file-collection", response_model=List[schemas.ShowFileCollection])
async def get_file_collection(db: Session = Depends(get_db), current_user: schemas.User = Depends(get_current_user)):
    files = db.query(models.FileCollection).all()
    return files

# Get data by user id
@app.get("/file-collection/{user_id}", response_model=List[schemas.ShowFileCollection])
async def get_file_collection_by_id(user_id: int, db: Session = Depends(get_db), current_user: schemas.User = Depends(get_current_user)):
    files = db.query(models.FileCollection).filter(models.FileCollection.user_id == user_id).all()
    
    requests = db.query(models.FileRequest).filter(and_(models.FileRequest.sender == current_user['id'], models.FileRequest.status == 2)).all()
    req_ids = []
    access_files = []
    for req in requests:
        req_ids.append(req.file_id)
    allFiles = db.query(models.FileCollection).all()
    
    for file in allFiles:
        if(file.id in req_ids):
            access_files.append(file)
    combinedFiles = files + access_files
    return combinedFiles

@app.get("/file-read/{file_id}")
async def file_read(file_id:int, db: Session = Depends(get_db), current_user: schemas.User = Depends(get_current_user)):
    file = db.query(models.FileCollection).filter(models.FileCollection.id == file_id).first()
    print(file)
    if(file.user_id == current_user['id']):
        file_location = os.path.join(f"../{current_user['id']}_stored_files/", file.filename)
    else: 
        file_location = os.path.join(f"../{file.user_id}_stored_files/", file.filename)
    jsondict = {}  

    with open(f"{file_location}", encoding="latin-1") as csvfile:
        csv_data = csv.DictReader(csvfile)
        jsondict["data"]=[]
        for rows in csv_data:
            jsondict["data"].append(rows)
    return jsondict

@app.get("/file_apply_model/{file_id}/{model}")
async def showresult(file_id: int, model: int, db: Session = Depends(get_db), current_user: schemas.ShowUser = Depends(get_current_user)):
    _models = [get_model_1, get_model_2, get_model_3]
    get_models = {i: model for i, model in enumerate(_models, 1)}
    # path = f"../{current_user['id']}_stored_files/{filename}"

    file = db.query(models.FileCollection).filter(models.FileCollection.id == file_id).first()
    if(file.user_id == current_user['id']):
        file_location = os.path.join(f"../{current_user['id']}_stored_files/", file.filename)
    else: 
        file_location = os.path.join(f"../{file.user_id}_stored_files/", file.filename)

    model_output = get_models[model](file_location)
    return model_output
    

@app.get("/file_access_for_request/{user_id}")
async def file_access_for_request(user_id:int, db: Session = Depends(get_db), current_user: schemas.User = Depends(get_current_user)):
    req_history = []

    user_files = db.query(models.FileCollection).filter(models.FileCollection.user_id == user_id).all()
    requests = db.query(models.FileRequest).filter(and_(models.FileRequest.sender == current_user['id'], models.FileRequest.reciever == user_id)).all()

    req_files_id = []
    for req in requests:
        req_files_id.append(req.file_id);

    for uf in user_files:
        if(uf.id in req_files_id):
            request = list(filter(lambda req : req.file_id == uf.id, requests))[0]
            req_history.append({
                'id': request.id,
                'file_id': uf.id,
                'file_name': uf.filename,
                'file_size': uf.filesize,
                'status': request.status,
                'file_owner': user_id
            })
        else:
            req_history.append({
                'id': 0,
                'file_id': uf.id,
                'file_name': uf.filename,
                'file_size': uf.filesize,
                'status': 0,
                'file_owner': user_id
            })
    return req_history

@app.post("/file_access_request_send")
async def file_access_request_send(request: schemas.FileRequestSend, db: Session = Depends(get_db), current_user: schemas.User = Depends(get_current_user)):
    file_req = models.FileRequest(file_id=request.file_id,sender=current_user['id'], reciever=request.file_owner, status= 1)
    db.add(file_req)
    db.commit()
    db.refresh(file_req)
    return file_req

@app.get("/file_access_list_for_accept_decline")
async def file_request_history(db: Session = Depends(get_db), current_user: schemas.User = Depends(get_current_user)):
    requests = db.query(models.FileRequest).filter(and_(models.FileRequest.status == 1, models.FileRequest.reciever == current_user['id'])).all()
    return requests

@app.get("/file_access_list_for_remove")
async def file_request_history(db: Session = Depends(get_db), current_user: schemas.User = Depends(get_current_user)):
    requests = db.query(models.FileRequest).filter(and_(models.FileRequest.status == 2, models.FileRequest.reciever == current_user['id'])).all()
    return requests

@app.delete("/file_access_remove_decline/{file_request_id}")
async def file_access_remvove_decline(file_request_id: int,db: Session = Depends(get_db), current_user: schemas.User = Depends(get_current_user)):
    db.query(models.FileRequest).filter(models.FileRequest.id == file_request_id).delete()
    db.commit()
    return "File Request Removed"

@app.patch("/file_access_approved/{file_request_id}")
async def file_access_remvove_decline(file_request_id: int, request: schemas.FileRequest, db: Session = Depends(get_db), current_user: schemas.User = Depends(get_current_user)):
    db.query(models.FileRequest).filter(models.FileRequest.id == file_request_id).update({
        'file_id': request.fileid,
        'sender': request.sender,
        'reciever': request.reciever,
        'status': 2
    });
    db.commit()
    return "Updated"

def run_training(id: int, job_id: str, name: str, ten_name: str, db: Session = Depends(get_db)):
    # Build fl training
    server: tuple = ()
    try:
        server = docker_cli.images.build(path=os.getcwd() + '/flwr_docker/server', tag=name, forcerm=True)
    except BuildError as e:
        logging.getLogger('uvicorn.error').error(e)
        db.query(models.LearningModel).filter(models.LearningModel.image_name==name).update({models.LearningModel.image_name: None})
        db.commit()
        return
    except APIError as e:
        logging.getLogger('uvicorn.error').error(e)
        db.query(models.LearningModel).filter(models.LearningModel.image_name==name).update({models.LearningModel.image_name: None})
        db.commit()
        return
    con_id: str = ""
    server_port: int = 0
    file_path = os.getcwd() + '/trainings/' + job_id
    try:
        os.mknod(file_path + '/clients')
        server_port = get_free_port()
        container = docker_cli.containers.run(image=server[0].id, name=name, detach=True, ports={8080: server_port}, volumes={file_path + '/clients': {'bind': '/server/clients', 'mode': 'rw'}, file_path + '/global_model': {'bind': '/server/global_model', 'mode': 'rw'}, file_path + '/flwr_logs': {'bind': '/server/flwr_logs', 'mode': 'rw'}, '/etc/localtime': {'bind': '/etc/localtime', 'mode': 'ro'}})
        con_id = container.id
    except ContainerError as e:
        logging.getLogger('uvicorn.error').error(e)
        return
    except ImageNotFound as e:
        logging.getLogger('uvicorn.error').error(e)
        return
    except APIError as e:
        logging.getLogger('uvicorn.error').error(e)
        return
    # Build tensorboard
    ten: tuple = ()
    try:
        ten = docker_cli.images.build(path=os.getcwd() + '/flwr_docker/tensorboard', tag=ten_name, forcerm=True)
    except BuildError as e:
        logging.getLogger('uvicorn.error').error(e)
        db.query(models.LearningModel).filter(models.LearningModel.tensorboard_image==ten_name).update({models.LearningModel.tensorboard_image: None})
        db.commit()
        return
    ten_id: str = ""
    ten_port: int = 0
    try:
        ten_port = get_free_port()
        con = docker_cli.containers.run(image=ten[0].id, name=ten_name, detach=True, ports={8090: ten_port}, volumes={file_path + '/flwr_logs': {'bind': '/logs', 'mode': 'ro'}, '/etc/localtime': {'bind': '/etc/localtime', 'mode': 'ro'}})
        ten_id = con.id
    except ContainerError as e:
        logging.getLogger('uvicorn.error').error(e)
        return
    except ImageNotFound as e:
        logging.getLogger('uvicorn.error').error(e)
        return
    except APIError as e:
        logging.getLogger('uvicorn.error').error(e)
        return 
    # Build global model download folder
    gmod_id: str = ""
    gmod_port: int = 0
    try:
        gmod_port = get_free_port()
        con = docker_cli.containers.run(image="nginx:stable", name='global_model_' + name, detach=True, ports={80: gmod_port}, volumes={file_path + '/global_model': {'bind': '/usr/share/nginx/global_model', 'mode': 'ro'}, os.getcwd() + '/flwr_docker/global_model/global_model.conf': {'bind': '/etc/nginx/nginx.conf', 'mode': 'ro'}, '/etc/localtime': {'bind': '/etc/localtime', 'mode': 'ro'}})
        gmod_id = con.id
    except ContainerError as e:
        logging.getLogger('uvicorn.error').error(e)
        return
    except ImageNotFound as e:
        logging.getLogger('uvicorn.error').error(e)
        return
    except APIError as e:
        logging.getLogger('uvicorn.error').error(e)
        return 
    db.query(models.LearningModel).filter(models.LearningModel.id==id).update({models.LearningModel.container_id: con_id, models.LearningModel.port: server_port, models.LearningModel.tensorboard_container: ten_id, models.LearningModel.tensorboard_port: ten_port, models.LearningModel.global_model_container: gmod_id, models.LearningModel.global_model_port: gmod_port})
    db.commit()

@app.post('/start/training/server')
def start_training_server(request: schemas.FLModel, background_tasks: BackgroundTasks, db: Session = Depends(get_db), current_user: schemas.User = Depends(get_current_user)):
    name = current_user['name'].strip().lower() + '_' + request.task.strip().lower() + '_' + str(hash(time()))
    ten_name = 'tensorboard_' + name
    job_id = str(uuid.uuid5(uuid.NAMESPACE_URL, name))
    # build and run training 
    file2zip(os.getcwd() + f'/downloads/{current_user["id"]}_stored_files/{job_id}.zip', [os.getcwd() + '/flwr_docker/run.sh', os.getcwd() + '/flwr_docker/run.ps1', os.getcwd() + '/flwr_docker/client/Dockerfile', os.getcwd() + '/flwr_docker/client/client.py'])
    link = f'/download/{current_user["id"]}/{job_id}.zip'
    new_fl = models.LearningModel(owner_id=current_user['id'], job_id=job_id, task=request.task.strip(), epochs=request.epochs, model=request.model.strip(), dataset=request.dataset.strip(), aggregation_approach=request.appr.strip(), port=None, address=get_host_ip(), image_name=name, container_id=None, link=link, tensorboard_port=None, tensorboard_image=ten_name, tensorboard_container=None, global_model_port=None, global_model_container=None)
    db.add(new_fl)
    db.commit()
    db.refresh(new_fl)
    os.makedirs(os.getcwd() + '/trainings/' + job_id)
    background_tasks.add_task(run_training, new_fl.id, job_id, name, ten_name, db)
    return JSONResponse(content={
        'id': new_fl.id,
        'job_id': job_id,
        'task': request.task.strip(),
        'rounds': request.epochs,
        'appr': request.appr.strip(),
        'status': 'building',
        'global_model': request.model.strip(),
        'metrics': request.dataset.strip(),
    }, status_code=status.HTTP_200_OK)

@app.get('/download/{id}/{file}')
async def download(id, file, current_user: schemas.User = Depends(get_current_user)):
    # return FileResponse(os.getcwd() + '/downloads/' + file, filename=file, background=BackgroundTask(lambda: os.remove(os.getcwd() + '/downloads/' + file)))
    path = os.getcwd() + '/downloads/' + id + '_stored_files/' + file
    if not os.path.isfile(path):
        return Response(status_code=status.HTTP_400_BAD_REQUEST, content='No such file')
    return FileResponse(path, filename=file)

# Get training status by id
@app.get('/training/status/{id}')
def training_status(id, db: Session = Depends(get_db), current_user: schemas.User = Depends(get_current_user)):
    c = db.query(models.LearningModel.container_id).filter(models.LearningModel.id==id).first()
    cid = c.container_id
    if cid is None:
        return JSONResponse(content={'code': 21000, 'status': 'No existed training'})
    status: str = ''
    try:
        con = docker_cli.containers.get(cid)
        status = con.attrs.get('State')['Status']
    except APIError as e:
        logging.getLogger('uvicorn.error').error(e)
        return Response(status_code=400)
    return JSONResponse(content={'code': 20000, 'status': status})

# Get all available training status
@app.get('/training/all/status')
def training_all_status(db: Session = Depends(get_db), current_user: schemas.User = Depends(get_current_user)):
    sets = db.query(models.LearningModel.id, models.LearningModel.container_id).filter(models.LearningModel.owner_id==current_user["id"]).all()
    ret: list = []
    for elem in sets:
        if elem.container_id is None:
            continue
        try:
            con = docker_cli.containers.get(elem.container_id)
            ret.append({'id': elem.id, 'status': con.attrs.get('State')['Status']})
        except APIError as e:
            logging.getLogger('uvicorn.error').error(e)
    return ret

# Access to clients' ip addresses
@app.get('/training/connections/{id}')
def training_connections(id, db: Session = Depends(get_db), current_user: schemas.User = Depends(get_current_user)):
    name = db.query(models.LearningModel.job_id).filter(models.LearningModel.id==id).first()
    global ip_url
    ip_url = 'http://ip-api.com/json/'
    server_location = requests.get(ip_url + get_host_ip()).json()
    sev_ret = {'status': server_location['status']}
    if server_location['status'] == 'success':
        sev_ret['country'] = server_location['country']
        sev_ret['city'] = server_location['city']
    else:
        sev_ret['message'] = server_location['message']
    resp = {
        'server': sev_ret,
        'client': []
    }
    with open(os.getcwd() + '/training/' + name + '/clients', 'r') as f:
        ips = f.read().split('\n')
        for elem in ips:
            cli_ret = requests.get(ip_url + elem.strip()).json()
            ret = {'status': cli_ret['status']}
            if cli_ret['status'] == 'success':
                ret['country'] = cli_ret['country']
                ret['city'] = cli_ret['city']
            else:
                ret['message'] = cli_ret['message']
            resp['client'].append(ret)
    return JSONResponse(resp)

# Delete fl training including container, image and data storaged in database
@app.delete('/training/delete/{id}')
def delete_training(id, db: Session = Depends(get_db), current_user: schemas.ShowUser = Depends(get_current_user)):
    img = db.query(models.LearningModel.job_id, models.LearningModel.image_name, models.LearningModel.container_id, models.LearningModel.tensorboard_image, models.LearningModel.tensorboard_container, models.LearningModel.global_model_container).filter(models.LearningModel.id==id, models.LearningModel.owner_id==current_user["id"]).first()
    if img is None:
        return Response(status_code=status.HTTP_400_BAD_REQUEST, content='No such training found')
    # Delete container and image for tensorboard
    tcid = img.tensorboard_container
    tname = img.tensorboard_image
    if tcid or tname:
        try:
            con = docker_cli.containers.get(tcid or tname)
            con.stop()
            con.remove(force=True)
        except NotFound as e:
            logging.getLogger('uvicorn.error').error(e)
        except APIError as e:
            logging.getLogger('uvicorn.error').error(e)
    if tname:
        try:
            docker_cli.images.remove(tname)
        except ImageNotFound as e:
            logging.getLogger('uvicorn.error').error(e)
    # Delete container for global models
    gmcid = img.global_model_container
    if gmcid:
        try:
            con = docker_cli.containers.get(gmcid)
            con.stop()
            con.remove(force=True)
        except NotFound as e:
            logging.getLogger('uvicorn.error').error(e)
        except APIError as e:
            logging.getLogger('uvicorn.error').error(e)
    # Delete container and image for training
    cid = img.container_id
    name = img.image_name
    if cid or name:
        try:
            con = docker_cli.containers.get(cid or name)
            con.stop()
            con.remove(force=True, v=True)
        except NotFound as e:
            logging.getLogger('uvicorn.error').error(e)
        except APIError as e:
            logging.getLogger('uvicorn.error').error(e)
    if name:
        try:
            docker_cli.images.remove(name)
        except ImageNotFound as e:
            logging.getLogger('uvicorn.error').error(e)
    # Delete files
    job_id = img.job_id
    try:
        os.remove(os.getcwd() + f'/downloads/{current_user["id"]}_stored_files/{job_id}.zip')
    except OSError as e:
        logging.getLogger('uvicorn.error').error(e)
    try:
        shutil.rmtree(os.getcwd() + f'/trainings/{job_id}')
    except OSError as e:
        logging.getLogger('uvicorn.error').error(e)
    # Delete row in database
    db.query(models.LearningModel).filter(models.LearningModel.id==id).delete()
    db.commit()
    return 'Training is deleted'

# Get a fl training by id
@app.get('/training/{id}')
def get_training_by_id(id, db: Session = Depends(get_db), current_user: schemas.ShowUser = Depends(get_current_user)):
    fl = db.query(models.LearningModel).filter(models.LearningModel.id==id, models.LearningModel.owner_id==current_user["id"]).first()
    if fl is None:
        return Response(status_code=400, content='No such training found')
    status: str = ''
    try:
        con = docker_cli.containers.get(fl.container_id)
        status = con.attrs.get('State')['Status']
    except APIError as e:
        logging.getLogger('uvicorn.error').error(e)
        status = 'fail'
    return JSONResponse(content={
        'id': fl.id,
        'job_id': fl.job_id,
        'task': fl.task,
        'rounds': fl.epochs,
        'appr': fl.aggregation_approach,
        'global_model': fl.model,
        'metrics': fl.dataset,
        'status': status,
        'download': fl.link,
        'ip': fl.address,
        'port': fl.port,
        'tensorboard_port': fl.tensorboard_port,
        'global_model_port': fl.global_model_port
    }, status_code=200)

# Get all the fl training for current user
@app.get('/trainings')
def get_all_trainings(db: Session = Depends(get_db), current_user: schemas.ShowUser = Depends(get_current_user)):
    sets = db.query(models.LearningModel).filter(models.LearningModel.owner_id==current_user['id']).all()
    ret: list = []
    for elem in sets:
        status: str = ''
        if elem.container_id is not None:
            try:
                con = docker_cli.containers.get(elem.container_id)
                status = con.attrs.get('State')['Status']
            except APIError as e:
                logging.getLogger('uvicorn.error').error(e)
                status = 'fail'
        else:
            status = 'none'
        item = {
            'id': elem.id,
            'job_id': elem.job_id,
            'task': elem.task,
            'rounds': elem.epochs,
            'appr': elem.aggregation_approach,
            'global_model': elem.model,
            'metrics': elem.dataset,
            'status': status,
            'download': elem.link,
            'ip': elem.address,
            'port': elem.port,
            'tensorboard_port': elem.tensorboard_port,
            'global_model_port': elem.global_model_port
        }
        ret.append(item)
    return ret 

# Upload new dataset
@app.post('/upload/dataset')
async def uploade_dataset(dataset: str = Form(), desc: str = Form(), affil: str = Form(), file_type: str = Form(), raw_file: UploadFile = File(), datatype: str = Form(), db: Session = Depends(get_db), current_user: schemas.ShowUser = Depends(get_current_user)):
    fs = await raw_file.read()
    filename = current_user["name"] + '_' + str(hash(time())) + '_' + raw_file.filename
    new_file = models.FileCollection(filename=filename, filesize=f'{len(fs)/1000} kb', user_id=current_user["id"])
    # db.new(new_file)
    db.add(new_file)
    db.commit()
    db.refresh(new_file)

    file_location = os.path.join(os.getcwd() + f"/downloads/{current_user['id']}_stored_files/", filename)
    with open(file_location, 'wb+') as file_obj:
        file_obj.write(fs)

    new_data = models.Dataset(dataset=dataset, owner_id=current_user["id"], description=desc, affiliation=affil, filetype=file_type.strip().lower(), filepath=new_file.id, datatype=datatype.strip().upper())
    # db.new(new_data)
    db.add(new_data)
    db.commit()
    db.refresh(new_data)

    return "Success to upload dataset"

# Get all the dataset for current user
@app.get('/datasets')
def get_datasets(db: Session = Depends(get_db), current_user: schemas.ShowUser = Depends(get_current_user)):
    sets = db.query(models.Dataset.id, models.Dataset.dataset, models.Dataset.description, models.Dataset.affiliation, models.Dataset.filetype, models.Dataset.datatype).filter(models.Dataset.owner_id==current_user['id']).all()
    return sets

# Get all datasets
@app.get('/datasets/all')
def get_all_datasets(db: Session = Depends(get_db), current_user: schemas.ShowUser = Depends(get_current_user)):
    sets = db.query(models.Dataset.id, models.Dataset.dataset, models.Dataset.description, models.Dataset.affiliation, models.Dataset.filetype, models.Dataset.datatype, models.User.name).join(models.User, models.Dataset.owner_id==models.User.id).all()
    return sets

# Get dataset by id
@app.get('/dataset/{id}')
def get_dataset_by_id(id, db: Session = Depends(get_db), current_user: schemas.ShowUser = Depends(get_current_user)):
    sql = f'select dataset, description, affiliation, filetype, datatype, name, if(users.id={current_user["id"]}, concat("/download/", users.id, "_stored_files/", filename), "url to application") as url from datasets, users, file_collection where datasets.owner_id = users.id and datasets.filepath = file_collection.id and datasets.id = {id};'
    data = db.execute(sql).fetchone()
    return data if data is not None else Response(status_code=400, content='No Existed Dataset')

# Delete dataset
@app.delete('/dataset/delete/{id}')
def delete_dataset(id, db: Session = Depends(get_db), current_user: schemas.ShowUser = Depends(get_current_user)):
    uid = current_user['id']
    data = db.query(models.Dataset).filter(models.Dataset.id==id).first()
    if data is None:
        return Response(status_code=400, content='No Existed Dataset')
    fid = data.filepath
    f = db.query(models.FileCollection).filter(models.FileCollection.id==fid).first()
    if f is None:
        return Response(status_code=400, content='No Such File')
    filename = f.filename
    filepath = os.getcwd() + f'/downloads/{uid}_stored_files/' + filename
    # Delete dataset
    db.query(models.Dataset).filter(models.Dataset.id==id).delete()
    db.commit()
    # Delete file
    db.query(models.FileCollection).filter(models.FileCollection.id==fid).delete()
    db.commit()
    if os.path.isfile(filepath):
        os.remove(filepath)
    return "Deleted dataset"