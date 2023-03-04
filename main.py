from turtle import st

from fastapi import FastAPI, UploadFile, File
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
import functools

from fastapi.responses import Response
from random import randint
import uuid
from PIL import Image


app = FastAPI()

origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

THIS_FILE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(THIS_FILE_DIR, 'models')
MODEL_FILENAME = os.path.join(MODEL_DIR, 'tensorflow_inception_graph.pb')

t_input = tf.compat.v1.placeholder(np.float32, name='input')


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


db = []
dbimg = []
def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))

@app.post("/images/")
async def create_upload_file(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()  # <-- Important!
    image = load_image_into_numpy_array(contents)

    arr = np.uint8(np.clip(image / 255.0, 0, 1) * 255)

    dbimg.append(arr)
    db.append(contents)

    return {"filename": file.filename}




@app.get("/images/")
async def read_random_file():
    random_index = randint(0, len(db) - 1)
    response = Response(content=db[random_index])
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

    return {"message": f"This is file which is came {file.filename}"}


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


















# @app.get("/resume/{id}")
# async def resume(id: str):
#     return {"message": f"This is  {id}'s resume"}
#
#
# @app.post('/file/upload')
# def upload_file(file: UploadFile):
#     if (file.content_type != 'application/json'):
#         raise HTTPException(400, detail="This is invalid type")
#     else:
#         data = json.loads(file.file.read())
#         return {"content": data, "filename": file.filename}



