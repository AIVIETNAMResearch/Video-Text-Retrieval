
from flask import Flask, render_template, Response, request, send_file, jsonify
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pandas as pd
import json
from pathlib import Path
from posixpath import join

from utils.faiss_processing import MyFaiss
from utils.submit import write_csv, show_csv
from utils.bert_processing import BERTSearch
from utils.ocr_processing import fill_ocr_results, fill_ocr_df

# http://0.0.0.0:5001/thumbnailimg?index=0

# app = Flask(__name__, template_folder='templates', static_folder='static')
app = Flask(__name__, template_folder='templates')

# Faiss
bin_file='dict/faiss_cosine.bin'
json_path = 'dict/keyframes_id.json'
json_id2img_path = 'dict/dict_id2img_path.json'
json_img2id_path = 'dict/dict_img2id_path.json'
json_keyframe2id = 'dict/keyframe_path2id.json'

with open("dict/info_ocr.txt", "r", encoding="utf8") as fi:
    ListOcrResults = list(map(lambda x: x.replace("\n",""), fi.readlines()))

with open("dict/info_asr.txt", "r", encoding="utf8") as fi: 
    ListASRResults = list(map(lambda x: x.replace("\n",""), fi.readlines()))
df_asr = pd.read_csv("dict/info_asr.txt", delimiter=",", header=None)
df_asr.columns = ["video_id", "frame_id", "asr"]    
        
with open(json_id2img_path, 'r') as f:
    DictId2Img = json.loads(f.read())

with open(json_img2id_path, 'r') as f:
    DictImg2Id = json.loads(f.read())

with open(json_keyframe2id, 'r') as f:
    DictKeyframe2Id = json.loads(f.read())

CosineFaiss = MyFaiss('Database', bin_file, json_path)
DictImagePath = CosineFaiss.id2img_fps
LenDictPath = len(CosineFaiss.id2img_fps)
print("LenDictPath: ", LenDictPath)
# CosineFaiss.id2img_fps

# BERT
MyBert = BERTSearch(dict_bert_search='dict/keyframes_id_bert.json', bin_file='dict/faiss_bert.bin', mode='search')

@app.route('/thumbnailimg')
def thumbnailimg():
    print("load_iddoc")

    # remove old file submit 
    submit_path = join("submission", "submit.csv")
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

@app.route('/asrsearch')
def asrsearch():
    print("asr search")
     # remove old file submit 

    pagefile = []
    text_query = request.args.get('text_asr')
    _, list_ids, _, list_image_paths = MyBert.bert_search(text_query, k=100)

    imgperindex = 100 

    for imgpath, id in zip(list_image_paths, list_ids):
        imgpath = imgpath.replace("\\","/")
        pagefile.append({'imgpath': imgpath, 'id': int(DictKeyframe2Id[imgpath])})

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    
    return render_template('index_thumb.html', data=data)

@app.route('/ocrfilter')
def ocrfilter():
    print("ocr search")

    pagefile = []
    text_query = request.args.get('text_ocr')

    list_all = fill_ocr_results(text_query, ListOcrResults)
    list_all.extend(fill_ocr_results(text_query, ListASRResults))

    # list_all = fill_ocr_df(text_query, df_ocr)
    # list_all = np.vstack((list_all, fill_ocr_df(text_query, df_ocr)))
    
    print("list results of ocr + asr: ", list_all)

    imgperindex = 100 

    for frame in list_all:
        list_frame_name = frame.split("/")
        keyframe_dir = list_frame_name[0][:7]
        video_dir = list_frame_name[0]
        new_frame_name = list_frame_name[-1]
        frame_in_video_path =  join("Database", "KeyFrames"+keyframe_dir, video_dir, new_frame_name)
        frame_in_video_path =  frame_in_video_path.replace("\\","/")
        # print("frame_in_video_path: ", frame_in_video_path)
        if frame_in_video_path in DictKeyframe2Id:
            print("frame_in_video_path: ", frame_in_video_path)
            frame_id_in_video_path = DictKeyframe2Id[frame_in_video_path]
            pagefile.append({'imgpath': frame_in_video_path, 'id': int(frame_id_in_video_path)})

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
        pagefile.append({'imgpath': shot_info['shot_path'], 'id': int(DictKeyframe2Id[shot_info['shot_path']])})

    # show  around 200 key image
    frame_path = DictImagePath[id_query]["image_path"]
    list_split = frame_path.split("/")
    keyframe_dir = list_split[1][-7:]
    video_dir = list_split[2]
    image_name = list_split[3]


    total_image_in_video = int(DictImg2Id[keyframe_dir][video_dir]["total_image"])
    number_image_id_in_video = int(DictImg2Id[keyframe_dir][video_dir][image_name])

    first_index_in_video = number_image_id_in_video-40 if number_image_id_in_video-40>0 else 0
    last_index_in_video = number_image_id_in_video+40 if number_image_id_in_video+40<total_image_in_video else total_image_in_video
    frame_index = first_index_in_video
    while frame_index < last_index_in_video:
        new_frame_name = DictId2Img[keyframe_dir][video_dir][str(frame_index)]
        frame_in_video_path =  join("Database", "KeyFrames"+keyframe_dir, video_dir, new_frame_name)
        frame_in_video_path =  frame_in_video_path.replace("\\","/")
        if frame_in_video_path in DictKeyframe2Id:
            frame_id_in_video_path = DictKeyframe2Id[frame_in_video_path]
            pagefile.append({'imgpath': frame_in_video_path, 'id': int(frame_id_in_video_path)})

        frame_index += 1

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    
    return render_template('index_thumb.html', data=data)

@app.route('/writecsv')
def submit():
    print("writecsv")
    info_key = request.args.get('info_key')
    mode_write_csv = request.args.get('mode')
    print("info_key", info_key)
    print("mode: ", mode_write_csv)
    info_key = info_key.split(",")

    id_query = int(info_key[0])
    selected_image = info_key[1]
    
    number_line, list_frame_id = write_csv(DictImagePath, mode_write_csv, selected_image, id_query, "submission")
    
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
    list_image_name = fpath.split("/")
    image_name = "/".join(list_image_name[-2:])

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
    filename = request.args.get('filename')
    fpath = join("submission", filename)
    print("fpath", fpath)

    return send_file(fpath, as_attachment=True)

@app.route('/get_first_row')
def getFirstRowOfCsv():
    csv_path = "submission/submit.csv"
    result = {
        'video_id':"None",
        'frame_id':"None"
    }
    if os.path.exists(csv_path):
        lst_frame = show_csv(csv_path)[0]
        video_id, frame_id = lst_frame.split("/")[-2:]
        result["video_id"] = video_id
        result["frame_id"] = int(frame_id[:-4])

    return result

@app.route('/visualize')
def visualize():
    number_of_query = int(request.args.get('number_of_query'))
    csv_path = join("submission", "query-{}.csv".format(number_of_query))

    query_path = join("query","query-{}.txt".format(number_of_query))
    if os.path.exists(query_path):
        with open(query_path, "rb") as fi:
            query_content = fi.read().decode("utf-8").replace(" ","_")

    pagefile = []
    lst_frame = show_csv(csv_path)
    for frame_path in lst_frame:
        frame_id = DictKeyframe2Id[frame_path]
        pagefile.append({'imgpath': frame_path, 'id': int(frame_id)})
    if query_content is not None:
        data = {'num_page': 1, 'pagefile': pagefile, 'query': query_content}
    else:
        data = {'num_page': 1, 'pagefile': pagefile}

    return render_template('index_thumb.html', data=data)

@app.route('/search_image_path')
def search_image_path():
    pagefile = []
    frame_path = request.args.get('frame_path')
    list_frame_split = frame_path.split("/")

    video_dir = list_frame_split[0]
    image_name = list_frame_split[1] + ".jpg"
    keyframe_dir = video_dir[:-2]

    frame_path = join("Database", "KeyFrames"+keyframe_dir, video_dir, image_name)
    frame_path = frame_path.replace("\\","/")
    frame_id = DictKeyframe2Id[frame_path]
    
    imgperindex = 100 
    pagefile.append({'imgpath': frame_path, 'id': int(frame_id)})

    # show  around 30 key image
    total_image_in_video = int(DictImg2Id[keyframe_dir][video_dir]["total_image"])
    number_image_id_in_video = int(DictImg2Id[keyframe_dir][video_dir][image_name])

    first_index_in_video = number_image_id_in_video-40 if number_image_id_in_video-40>0 else 0
    last_index_in_video = number_image_id_in_video+40 if number_image_id_in_video+40<total_image_in_video else total_image_in_video

    frame_index = first_index_in_video
    while frame_index < last_index_in_video:
        new_frame_name = DictId2Img[keyframe_dir][video_dir][str(frame_index)]
        frame_in_video_path =  join("Database", "KeyFrames"+keyframe_dir, video_dir, new_frame_name)
        frame_in_video_path = frame_in_video_path.replace("\\","/")
        if frame_in_video_path in DictKeyframe2Id:
            frame_id_in_video_path = DictKeyframe2Id[frame_in_video_path]
            pagefile.append({'imgpath': frame_in_video_path, 'id': int(frame_id_in_video_path)})

        frame_index += 1

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    
    return render_template('index_thumb.html', data=data)


if __name__ == '__main__':
    submit_dir = "submission"
    if not os.path.exists(submit_dir):
        os.mkdir(submit_dir)

    app.run(debug=False, host="0.0.0.0", port=5001)
