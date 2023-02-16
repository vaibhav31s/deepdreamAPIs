from fastapi import FastAPI,UploadFile,File
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
import base64
from werkzeug.utils import secure_filename
from io import BytesIO
import PIL.Image
import numpy as np
import os
import json
import requests
import subprocess
import tensorflow as tf
import urllib
import tensorflow as tf
import numpy as np
import PIL.Image
import IPython.display as display
import zipfile
from fastapi.responses import FileResponse

from fastapi.responses import Response
from random import randint
import uuid

app = FastAPI()

origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ['*'],
    allow_headers = ['*']
)

THIS_FILE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(THIS_FILE_DIR, 'models')
MODEL_FILENAME = os.path.join(MODEL_DIR, 'tensorflow_inception_graph.pb')


def download_model_from_web():
    if os.path.isfile(MODEL_FILENAME):
        return

    try:
        os.mkdir(MODEL_DIR)
    except FileExistsError:
        pass

    MODEL_ZIP_URL = (
        'https://storage.googleapis.com/download.tensorflow.org/models/'
        'inception5h.zip')
    ZIP_FILE_NAME = 'inception5h.zip'
    ZIP_FILE_PATH = os.path.join(MODEL_DIR, ZIP_FILE_NAME)
    resp = requests.get(MODEL_ZIP_URL, stream=True)

    with open(ZIP_FILE_PATH, 'wb') as file_desc:
        for chunk in resp.iter_content(chunk_size=5000000):
            file_desc.write(chunk)

    zip_file = zipfile.ZipFile(ZIP_FILE_PATH)
    zip_file.extractall(path=MODEL_DIR)

    os.remove(ZIP_FILE_PATH)

def init_model():
    with tf.compat.v1.gfile.FastGFile(MODEL_FILENAME, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

download_model_from_web()
graph_def = init_model()

graph = tf.Graph()
sess = tf.compat.v1.InteractiveSession(graph=graph)






@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.get("/resume/{id}")
async def resume(id : str):
    return {"message" : f"This is  {id}'s resume"}


@app.post('/file/upload')
def upload_file(file: UploadFile):
    if(file.content_type != 'application/json'):
        raise HTTPException(400,detail="This is invalid type")
    else :
        data = json.loads(file.file.read())
        return {"content":data, "filename":file.filename}

db = []

@app.post("/images/")
async def create_upload_file(file: UploadFile = File(...)):
    
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()  # <-- Important!

    db.append(contents)

    return {"filename": file.filename}


@app.get("/images/")
async def read_random_file():
    random_index = randint(0, len(db) - 1)
    response = Response(content=db[random_index])
    # print(db)
    return response




@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)

    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"message": f"This is file which is came {file.filename}" }
# print(img)


from typing import List


@app.post("/files")
def uploadss(files: List[UploadFile] = File(...)):
    for file in files:
        try:
            with open(file.filename, 'wb') as f:
                while contents := file.file.read(1024 * 1024):
                    f.write(contents)
        except Exception:
            return {"message": "There was an error uploading the file(s)"}
        finally:
            file.file.close()

    return {"message": f"Successfuly uploaded {[file.filename for file in files]}"}



