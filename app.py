
from flask import Flask, render_template, jsonify, Response, request
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pandas as pd
import json
from utils.faiss_processing import MyFaiss

# http://0.0.0.0:5001/thumbnailimg?index=0

# app = Flask(__name__, template_folder='templates', static_folder='static')
app = Flask(__name__, template_folder='templates')

# Faiss
bin_file='dict/faiss_cosine.bin'
json_path = 'dict/keyframes_id.json'

CosineFaiss = MyFaiss('Database', bin_file, json_path)
DictImagePath = CosineFaiss.id2img_fps
LenDictPath = len(CosineFaiss.id2img_fps)
# CosineFaiss.id2img_fps

@app.route('/thumbnailimg')
def thumbnailimg():
    print("load_iddoc")
    pagefile = []
    index = int(request.args.get('index'))
    if index == None:
        index = 0

    imgperindex = 100
    
    # imgpath = request.args.get('imgpath') + "/"
    pagefile = []

    page_filelist = []
    list_idx = []

    if LenDictPath-1 > index+imgperindex:
        first_index = index * imgperindex
        last_index = index*imgperindex + imgperindex

        tmp_index = first_index
        while tmp_index < last_index:
            page_filelist.append(DictImagePath[tmp_index]["image_path"])
            list_idx.append(tmp_index)
            tmp_index += 1    
    else:
        first_index = index * imgperindex
        last_index = LenDictPath

        tmp_index = first_index
        while tmp_index < last_index:
            page_filelist.append(DictImagePath[tmp_index]["image_path"])
            list_idx.append(tmp_index)
            tmp_index += 1    

    for imgpath, id in zip(page_filelist, list_idx):
        pagefile.append({'imgpath': imgpath, 'id': id})

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    
    return render_template('index_thumb.html', data=data)

@app.route('/imgsearch')
def image_search():
    print("image search")
    pagefile = []
    id_query = int(request.args.get('imgid'))
    _, list_idx, list_image_paths = CosineFaiss.image_search(id_query, k=200)

    imgperindex = 100 

    for imgpath, id in zip(list_image_paths, list_idx):
        pagefile.append({'imgpath': imgpath, 'id': int(id)})

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    
    return render_template('index_thumb.html', data=data)

@app.route('/textsearch')
def text_search():
    print("text search")
    pagefile = []
    text_query = request.args.get('textquery')
    _, list_idx, list_image_paths = CosineFaiss.text_search(text_query, k=200)

    imgperindex = 100 

    for imgpath, id in zip(list_image_paths, list_idx):
        pagefile.append({'imgpath': imgpath, 'id': int(id)})

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    
    return render_template('index_thumb.html', data=data)


@app.route('/get_img')
def get_img():
    # print("get_img")
    fpath = request.args.get('fpath')
    fpath = fpath
    image_name = fpath.split("/")[-1]

    if os.path.exists(fpath):
        img = cv2.imread(fpath)
    else:
        print("load 404.jph")
        img = cv2.imread("./static/images/404.jpg")

    # print(img.shape)
    img = cv2.putText(img, image_name, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                   3, (255, 0, 0), 4, cv2.LINE_AA)

    ret, jpeg = cv2.imencode('.jpg', img)
    return  Response((b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)