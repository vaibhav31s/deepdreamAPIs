
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import PIL.Image
import os
import urllib
import tensorflow as tf
import numpy as np
import PIL.Image
from fastapi.responses import FileResponse
import functools

tf.compat.v1.disable_eager_execution()

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
    showImgInput(arr)
    dbimg.append(arr)
    db.append(contents)
    # print(contents.)

    return {"filename": file.filename}


@app.get("/images/")
async def read_random_file():
    # random_index = randint(0, len(db) - 1)
    response = Response(content=db[0])

    # showImg(db[0])
    return response


@app.get("/image")
async def inputImage():
    response = FileResponse("input.png")
    return response



# print(img)

@functools.lru_cache
def init_model():
    with tf.compat.v1.gfile.FastGFile(MODEL_FILENAME, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

graph_def = init_model()
graph = tf.Graph()
sess = tf.compat.v1.InteractiveSession(graph=graph)

t_input = tf.compat.v1.placeholder(np.float32, name='input')
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})

def get_tensor(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name('%s:0' % layer)
img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0

def write_image(dg, arr):

    arr = np.uint8(np.clip(arr/255.0, 0, 1)*255)
    dg.image(arr, use_column_width=True)
    return dg
def tffunc(*argtypes):
    placeholders = list(map(tf.compat.v1.placeholder, argtypes))

    def wrap(f):
        out = f(*placeholders)

        def wrapper(*args, **kw):
            return out.eval(
                dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.compat.v1.image.resize_bilinear(img, size)[0, :, :, :]

resize = tffunc(np.float32, np.int32)(resize)

def calc_grad_tiled(img, t_grad, tile_size=512):

    '''Compute the value of tensor t_grad over the image in a tiled way.

    Random shifts are applied to the image to blur tile boundaries over
    multiple iterations.
    '''
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h-sz//2, sz), sz):
        for x in range(0, max(w-sz//2, sz), sz):
            sub = img_shift[y:y+sz, x:x+sz]
            g = sess.run(t_grad, {t_input: sub})
            grad[y:y+sz, x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

ans = None

def showImg(img_in):
    arr = np.uint8(np.clip(img_in / 255.0, 0, 1) * 255)
    img = Image.fromarray(arr, 'RGB')
    img.save('output.png')
    # img.show()
    return img
def saveImage(img_in):
    arr = np.uint8(np.clip(img_in / 255.0, 0, 1) * 255)
    img = Image.fromarray(arr, 'RGB')
    img.save('output.png')
    # img.show()
    return img
def showImgInput(img_in):
    arr = np.uint8(np.clip(img_in / 255.0, 0, 1) * 255)
    img = Image.fromarray(arr, 'RGB')
    img.save('input.png')
    img.show()
    return img

def do_deepdream(
        t_obj, img_in=img_noise, iter_n=10, step=1.5, octave_n=4,
        octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]

    # split the image into a number of octaves
    octaves = []
    for i in range(octave_n-1):
        hw = img_in.shape[:2]
        lo = resize(img_in, np.int32(np.float32(hw)/octave_scale))
        hi = img_in-resize(lo, hw)
        img_in = lo
        octaves.append(hi)


    p = 0.0

    # generate details octave by octave
    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img_in = resize(img_in, hi.shape[:2])+hi
        for i in range(iter_n):
            g = calc_grad_tiled(img_in, t_grad)
            img_in += g*(step / (np.abs(g).mean()+1e-7))
            p += 1
            # showImg(img_in)
        saveImage(img_in)

    return showImg(img_in)

    # print(ans)

layers = [
    op.name for op in graph.get_operations()
    if op.type == 'Conv2D' and 'import/' in op.name
    ]


@functools.lru_cache
def read_file_from_url(url):
    return urllib.request.urlopen(url).read()


from pydantic import BaseModel
class outputParams(BaseModel):
    layers : int
    channel :int
    octaves:int
    iterations:int

@app.post("/out")
async def output(params : outputParams):
    MAX_IMG_WIDTH = 1200
    MAX_IMG_HEIGHT = 800
    DEFAULT_IMAGE_URL = './/'

    file_obj = 'input.png'

    img_in = PIL.Image.open(file_obj)
    img_in.thumbnail((MAX_IMG_WIDTH, MAX_IMG_HEIGHT), PIL.Image.ANTIALIAS)
    img_in = np.float32(img_in)
    max_value = len(layers) - 1
    layer_num = params.layers
    layer = layers[layer_num]

    channels = int(get_tensor(layer).get_shape()[-1])
    max_value = channels - 1
    channel = params.channel

    octaves = params.octaves

    iterations = params.iterations
    out = do_deepdream(
        get_tensor(layer)[:, :, :, channel], img_in, octave_n=octaves,
        iter_n=iterations)

    response = FileResponse("input.png")
    return response

@app.get("/out")
async def output():
    response = FileResponse("output.png")
    return response
















