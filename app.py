
from flask import Flask, render_template, Response, request, send_file, jsonify
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pandas as pd
import json
from pathlib import Path

from utils.faiss_processing import MyFaiss
from utils.submit import write_csv

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

    # remove old file submit 
    submit_path = os.path.join("submission", "submit.csv")
    old_submit_path = Path(submit_path)
    if old_submit_path.is_file():
        os.remove(submit_path)
        # open(submit_path, 'w').close()

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
    _, list_ids, _, list_image_paths = CosineFaiss.image_search(id_query, k=200)

    imgperindex = 100 

    for imgpath, id in zip(list_image_paths, list_ids):
        pagefile.append({'imgpath': imgpath, 'id': int(id)})

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    
    return render_template('index_thumb.html', data=data)

@app.route('/textsearch')
def text_search():
    print("text search")
    pagefile = []
    text_query = request.args.get('textquery')
    _, list_ids, _, list_image_paths = CosineFaiss.text_search(text_query, k=200)

    imgperindex = 100 

    for imgpath, id in zip(list_image_paths, list_ids):
        pagefile.append({'imgpath': imgpath, 'id': int(id)})

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    
    return render_template('index_thumb.html', data=data)

@app.route('/showsegment')
def show_segment():
    print("showsegment")
    pagefile = []
    id_query = int(request.args.get('imgid'))

    list_shot_path = DictImagePath[id_query]['list_shot_path']
    
    imgperindex = 100 
    for shot_info in list_shot_path:
        pagefile.append({'imgpath': shot_info['shot_path'], 'id': int(shot_info['shot_id'])})

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    
    return render_template('index_thumb.html', data=data)

@app.route('/writecsv')
def submit():
    print("writecsv")
    id_query = int(request.args.get('imgid'))
    
    number_line, list_frame_id = write_csv(DictImagePath, id_query, "submission")
    
    str_fname = ",".join(list_frame_id[:])
    # str_fname += " #### number csv line: {}".format(number_line)

    info = {
        "str_fname": str_fname,
        "number_line": str(number_line)
    }

    return jsonify(info)

@app.route('/get_img')
def get_img():
    # print("get_img")
    fpath = request.args.get('fpath')
    # fpath = fpath
    image_name = fpath.split("/")[-1]

    if os.path.exists(fpath):
        img = cv2.imread(fpath)
    else:
        print("load 404.jph")
        img = cv2.imread("./static/images/404.jpg")

    img = cv2.resize(img, (1280,720))

    # print(img.shape)
    img = cv2.putText(img, image_name, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                   3, (255, 0, 0), 4, cv2.LINE_AA)

    ret, jpeg = cv2.imencode('.jpg', img)
    return  Response((b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/dowload_submit_file', methods=['GET'])
def dowload_submit_file():
    print("dowload_submit_file")
    fpath = request.args.get('filepath')
    print("fpath", fpath)

    return send_file(fpath, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5001)