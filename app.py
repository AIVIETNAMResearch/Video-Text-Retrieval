from flask import Flask, render_template, Response, request, send_file, jsonify
import cv2
import os
import torch
import sys
import pandas as pd
import json
import faiss
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# from utils.nlp_processing import Translation
from utils.translate_vi2en import translate_vi2en
from pathlib import Path
from posixpath import join
from langdetect import detect
from utils.faiss_processing import write_csv, extract_feats_from_bin, save_feats_to_bin, get_id2index, \
    load_json_file,load_bin_file,mapping_index, search_tags, get_all_ids, remove_keys_and_save, search_image2image, \
    searchcontinues, read_index_file, save_bin_delete_noise
from utils.submit import write_csv, show_csv
# from sentence_transformers import SentenceTransformer, util
# from utils.ocr_processing import fill_ocr_results, fill_ocr_df

from utils.group_keyframes import convertArray 

current_dir = os.path.dirname(os.path.abspath(__file__))

# Xác định đường dẫn tới thư mục LAVIS
lavis_dir = os.path.join(current_dir, 'LAVIS')

# Thêm đường dẫn tương đối của thư mục LAVIS vào sys.path
sys.path.append(lavis_dir)
from lavis.models import load_model_and_preprocess
# http://0.0.0.0:5001/thumbnailimg?index=0

# app = Flask(__name__, template_folder='templates', static_folder='static')
app = Flask(__name__, template_folder='templates')

# Faiss
bin_file='dict/faiss_blip_v1_cosine_final.bin'
json_path = 'dict/dict_final/keyframes_id.json'
json_id2img_path = 'dict/dict_final/dict_image_path_id2img.json'
json_img2id_path = 'dict/dict_final/dict_image_path_img2id.json'
json_keyframe2id = 'dict/dict_final/keyframe_path2id.json'
json_keyframe2path = 'dict/dict_final/keyframe_id2path.json'
file_path = 'search_continues/list_index_search_continues.txt'

################# LOAD FILE BIN ##################        
faiss_model = load_bin_file(bin_file)

############## MODEL BLIP #################
__device = "cuda" if torch.cuda.is_available() else "cpu"
model, vis_processors_blip, text_processors_blip = load_model_and_preprocess("blip_image_text_matching", 
                                                                                      "base", 
                                                                                      device=__device, 
                                                                                      is_eval=True)
# with open("dict/info_ocr.txt", "r", encoding="utf8") as fi:
#     ListOcrResults = list(map(lambda x: x.replace("\n",""), fi.readlines()))

# with open("dict/info_asr.txt", "r", encoding="utf8") as fi: 
#     ListASRResults = list(map(lambda x: x.replace("\n",""), fi.readlines()))
# df_asr = pd.read_csv("dict/info_asr.txt", delimiter=",", header=None)
# df_asr.columns = ["video_id", "frame_id", "asr"]    
        
with open(json_id2img_path, 'r') as f:
    DictId2Img = json.loads(f.read())

with open(json_img2id_path, 'r') as f:
    DictImg2Id = json.loads(f.read())

with open(json_keyframe2path, 'r') as f:
    DictKeyframe2Path = json.loads(f.read())

with open(json_keyframe2id, 'r') as f:
    DictKeyframe2Id = json.loads(f.read())

LenDictPath = len(load_json_file(json_path))
DictImagePath = load_json_file(json_path)

######################### HOME PAGE ########################################
@app.route('/thumbnailimg')
def thumbnailimg():
    print("load_iddoc")
        
    # remove old file submit 
    submit_path = join("submission", "submit.csv")
    old_submit_path = Path(submit_path)
    
    if old_submit_path.is_file():
        os.remove(submit_path)
        # open(submit_path, 'w').close()
    temp_faiss_path = join("search_continues", "temp_faiss.bin")
    old_temp_path = Path(temp_faiss_path)
    if old_temp_path.is_file():
        os.remove(old_temp_path)
    
    temp_txt_path = join("search_continues", "list_index_search_continues.txt")
    old_temp_path = Path(temp_txt_path)
    if old_temp_path.is_file():
        os.remove(old_temp_path)
        
    objs_path = Path('search_continues/objs.csv')
    if objs_path.is_file():
        os.remove(objs_path)
        
    if os.path.exists('search_continues/after_rm_noise.bin'):
        os.remove('search_continues/after_rm_noise.bin')

    if os.path.exists('search_continues/after_rm_noise_idx.txt'):
        os.remove('search_continues/after_rm_noise_idx.txt')
    
    if os.path.exists('search_continues/after_dict_rm_noise.json'):
        os.remove('search_continues/after_dict_rm_noise.json')
        
    # bin_file = 'dict/faiss_blip_v1_cosine_final.bin'
    print("LenDictPath: ", LenDictPath)
    
    pagefile = []
    index = int(request.args.get('index'))
    if index == None:
        index = 0

    imgperindex = 100
    
    pagefile = []

    page_filelist = []
    list_idx = []
    # print(index)
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

    pagefile_new = convertArray(pagefile)
    
    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile_new}
    
    return render_template('home.html', data=data)

###################### SEARCH IMAGE PATH #####################
@app.route('/search_image_path')
def search_image_path():
    pagefile = []
    frame_path = request.args.get('frame_path')
    list_frame_split = frame_path.split("/")
    
    video_dir = list_frame_split[0]
    image_name = list_frame_split[1] + ".jpg"
    keyframe_dir = video_dir.split('_')[0]
    

    frame_path = join("Database", "Keyframes_"+keyframe_dir, video_dir, image_name)
    frame_path = frame_path.replace("\\","/")
    frame_id = DictKeyframe2Id[frame_path]
    
    imgperindex = 100 
    pagefile.append({'imgpath': frame_path, 'id': int(frame_id)})

    # show  around 40 key image
    total_image_in_video = int(DictImg2Id[keyframe_dir][video_dir]["total_image"])
    number_image_id_in_video = int(DictImg2Id[keyframe_dir][video_dir][image_name])

    first_index_in_video = number_image_id_in_video-50 if number_image_id_in_video-50>0 else 0
    last_index_in_video = number_image_id_in_video+50 if number_image_id_in_video+50<total_image_in_video else total_image_in_video
    frame_index = first_index_in_video
    while frame_index < last_index_in_video:
        new_frame_name = DictId2Img[keyframe_dir][video_dir][str(frame_index)]
        frame_in_video_path =  join("Database", "Keyframes_"+keyframe_dir, video_dir, new_frame_name)
        frame_in_video_path =  frame_in_video_path.replace("\\","/")
        if frame_in_video_path in DictKeyframe2Id:
            frame_id_in_video_path = DictKeyframe2Id[frame_in_video_path]
            pagefile.append({'imgpath': frame_in_video_path, 'id': int(frame_id_in_video_path)})

        frame_index += 1
    pagefile_new = convertArray(pagefile)

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile_new}
    
    return render_template('home.html', data=data)

####################### IMAGE SEARCH - SEARCH TO KEYFRAME ID ######################
import json

@app.route('/imgsearch')
def image_search():
    print("image search")
    pagefile = []
    id_query = int(request.args.get('imgid'))
    
    print('imgid: ', id_query)
    k = request.args.get('topk')
    k = int(k[3:])
    
    # Check search continues
    after_rm_noise_path = Path('search_continues/after_rm_noise.bin')
    faiss_path = Path('search_continues/temp_faiss.bin')
    
    if os.path.exists('search_continues/after_rm_noise_idx.txt'):
        index_file = 'search_continues/after_rm_noise_idx.txt'
        save_bin_delete_noise(bin_file, index_file, 'search_continues/after_rm_noise.bin')
    
    if after_rm_noise_path.is_file(): # Mức độ ưu tiên của file bin search: after_rm_noise --> temp_faiss --> file bin gốc
        print("continue searchingggg after delete noiseeee....................................")
        new_file_bin = 'search_continues/after_rm_noise.bin'
        index_file = 'search_continues/after_rm_noise_idx.txt'
        scores, idx_images = searchcontinues(new_file_bin, index_file, k, id_query=id_query)
    elif faiss_path.is_file():
        print("continue searchinggg...........................................................")
        new_file_bin = 'search_continues/temp_faiss.bin'
        index_file = "search_continues/list_index_search_continues.txt"
        scores, idx_images = searchcontinues(new_file_bin, index_file, k, id_query=id_query)
    else:
        query_feats = faiss_model.reconstruct(id_query).reshape(1,-1)
        scores, idx_images = faiss_model.search(query_feats, k=k)
        idx_images = idx_images.flatten()
        scores = scores.flatten()
        
    id2img_fps = DictImagePath
    infos_query = list(map(id2img_fps.get, list(idx_images)))
    image_paths = [info['image_path'] for info in infos_query]
    scores = np.array(scores, dtype=np.float32).tolist()
    
    imgperindex = 100 

    for imgpath, id, score in zip(image_paths, idx_images, scores):
        pagefile.append({'imgpath': imgpath, 'id': int(id), 'score':score})
    
    pagefile_new = convertArray(pagefile)
    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile_new}
    
    return render_template('index_thumb1.html', data=data)

######################### SEARCH TO TEXT - BLIP SEARCH ############################
@app.route('/textsearch')
def text_search():
    print("text search")
    k = str(request.args.get('topk'))
    k = int(k[3:])
    
    pagefile = []
    text_query = request.args.get('textquery')
    
    if detect(text_query) == 'vi':
        # translater = Translation()
        # text = translater(text_query)
        text = translate_vi2en(text_query)
    else:
        text = text_query

    ###### TEXT FEATURES EXACTING ######
    txt = text_processors_blip["eval"](text)
    text_features = model.encode_text(txt, __device).cpu().detach().numpy()
     
    ##### CHECK SEARCH CONTINUES #####
    after_rm_noise_path = Path('search_continues/after_rm_noise.bin')
    faiss_path = Path('search_continues/temp_faiss.bin')
    ###### SEARCHING #####
    if os.path.exists('search_continues/after_rm_noise_idx.txt'):
        index_file = 'search_continues/after_rm_noise_idx.txt'
        save_bin_delete_noise(bin_file, index_file, 'search_continues/after_rm_noise.bin')
        
    if after_rm_noise_path.is_file(): # Mức độ ưu tiên của file bin search: after_rm_noise --> temp_faiss --> file bin gốc
        print("continue searchingggg after delete noiseeee....................................")
        new_file_bin = 'search_continues/after_rm_noise.bin'
        index_file = 'search_continues/after_rm_noise_idx.txt'
        scores, idx_images = searchcontinues(new_file_bin, index_file, k, text_features=text_features)
    elif faiss_path.is_file():
        print("continue searchinggg...........................................................")
        new_file_bin = 'search_continues/temp_faiss.bin'
        index_file = "search_continues/list_index_search_continues.txt"
        scores, idx_images = searchcontinues(new_file_bin, index_file, k, text_features=text_features)
    else:
        scores, idx_images = faiss_model.search(text_features, k=k)
        idx_images = idx_images.flatten()
        scores = scores.flatten()

    ###### GET INFOS KEYFRAMES_ID ######
    id2img_fps = DictImagePath
    infos_query = list(map(id2img_fps.get, list(idx_images)))
    image_paths = [info['image_path'] for info in infos_query]
    
    imgperindex = 100 
    scores = np.array(scores, dtype=np.float32).tolist()
    # print(scores)

    for imgpath, id, score in zip(image_paths, idx_images, scores):
        pagefile.append({'imgpath': imgpath, 'id': int(id), 'score':score})
    pagefile_new = convertArray(pagefile)
    # print(pagefile_new)
    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile_new}
    
    return render_template('index_thumb1.html', data=data)

####################### CONTINUES SEARCHINGGGG ####################
@app.route('/searchcontinues', methods=['POST'])
def search_continues():
    data = request.get_json()
    pagefile = data['pagelist']
    
    # Sử dụng hàm để lấy danh sách tất cả các ID
    indexs = get_all_ids(pagefile)
    print(indexs)
    
    new_bin_file = './search_continues/temp_faiss.bin'
    new_list_idx_for_bin = './search_continues/list_index_search_continues.txt'
    # bin_file = 'dict/faiss_blip_v1_cosine_final.bin'
    
    if Path(new_list_idx_for_bin).is_file():
        data_array = read_index_file(new_list_idx_for_bin)
        ids = get_id2index(data_array) ## get index for index_list
        print(ids)
        feats = extract_feats_from_bin(bin_file, ids)
    else:
        feats = extract_feats_from_bin(bin_file, indexs)
    
    # savefile sub bin and idx of frames
    save_feats_to_bin(indexs, feats, new_bin_file, new_list_idx_for_bin)

    after_rm_noise_path = Path('search_continues/after_rm_noise.bin')
    after_rm_noise_txt_path = Path('search_continues/after_rm_noise_idx.txt')
    if after_rm_noise_txt_path.is_file():
        os.remove(after_rm_noise_txt_path)
    if after_rm_noise_path.is_file():
        os.remove(after_rm_noise_path)
        
    print('Saved new bin file')

######################### GET FRAMES NEIGHTBOR ##############################
@app.route('/neighborsearch')
def neightbor_search():
    print('neightbor frame search')
    pagefile = []
    id_query = int(request.args.get('imgid'))

    
    list_shot_path = DictImagePath[id_query]['list_shot_path']
    
    imgperindex = 100 
    for shot_info in list_shot_path:
        pagefile.append({'imgpath': shot_info['shot_path'], 'id': int(DictKeyframe2Id[shot_info['shot_path']])})

    # show  around 200 key image
    frame_path = DictImagePath[id_query]["image_path"]
    video_dir = frame_path.split("/")[-2]
    keyframe_dir = video_dir.split('_')[0]
    image_name = frame_path.split("/")[-1]


    total_image_in_video = int(DictImg2Id[keyframe_dir][video_dir]["total_image"])
    number_image_id_in_video = int(DictImg2Id[keyframe_dir][video_dir][image_name])

    first_index_in_video = number_image_id_in_video-50 if number_image_id_in_video-50>0 else 0
    last_index_in_video = number_image_id_in_video+50 if number_image_id_in_video+50<total_image_in_video else total_image_in_video
    frame_index = first_index_in_video
    while frame_index < last_index_in_video:
        new_frame_name = DictId2Img[keyframe_dir][video_dir][str(frame_index)]
        frame_in_video_path =  join("Database", "Keyframes_"+keyframe_dir, video_dir, new_frame_name)
        frame_in_video_path =  frame_in_video_path.replace("\\","/")
        if frame_in_video_path in DictKeyframe2Id:
            frame_id_in_video_path = DictKeyframe2Id[frame_in_video_path]
            pagefile.append({'imgpath': frame_in_video_path, 'id': int(frame_id_in_video_path)})

        frame_index += 1
    pagefile_new = convertArray(pagefile)
    # print(pagefile_new)
    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile_new}
    
    return render_template('home.html', data=data)

####################### SEARCH FOR TAGS - SEARCH OBJ #####################    
@app.route('/search_for_tags')
def search_for_tags():
    print("search for tags...")
    k = str(request.args.get('topk'))
    k = int(k[3:])

    pagefile = []
    text_query = request.args.get('text_for_tags')

    if Path('search_continues/objs.csv').is_file():
        csv_file = 'search_continues/objs.csv'
    else:
        csv_file = 'dict/object_final.csv'
        
    text_query = str(text_query)
    print(text_query)
    
    objs, idx_images, image_paths = search_tags(csv_file, text_query)
    objs.to_csv('search_continues/objs.csv', index=False)  # index=False để không lưu cột index
    
    imgperindex = 100 

    for imgpath, id in zip(image_paths, idx_images):
        pagefile.append({'imgpath': imgpath, 'id': int(id)})
    
    pagefile_new = convertArray(pagefile)

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile_new}
    
    return render_template('index_thumb1.html', data=data)

########################## DELETE VIDEO ID NOISE #########################
@app.route('/delete_noise', methods=['POST'])
def delete_noise():
    # Nhận dữ liệu từ yêu cầu POST
    video_id = request.form.get('video_id')
    print(video_id)
    list_frame_json = request.form.get('list_frame')

    # Tiến hành xóa nhiễu hoặc thực hiện các tác vụ khác ở đây
    # new_bin_path = 'search_continues/after_rm_noise.bin'
    new_list_idx_path = 'search_continues/after_rm_noise_idx.txt'
    new_dict_path = 'search_continues/after_dict_rm_noise.json'
    
    if os.path.exists(new_dict_path):
        ids, new_dict = remove_keys_and_save(video_id, new_dict_path)
    else:
        ids, new_dict = remove_keys_and_save(video_id, json_keyframe2id)
        
    with open(new_list_idx_path, 'w') as output_file:
        json.dump(ids, output_file)
    with open(new_dict_path, 'w') as output:
        json.dump(new_dict, output)
    # bin_file = 'dict/faiss_blip_v1_cosine_final.bin'
    # ids, feats = extract_feats_from_bin(bin_file, ids)
    
    # # savefile sub bin and idx of frames
    # index = faiss.IndexFlatIP(256)
    # index.add(feats)
    # faiss.write_index(index, new_bin_path)
    # print('Saved new bin file')
    

    # Trả về phản hồi (nếu cần)
    return 'Success: Saved delete noiseee!!!', 200

########################## WRITE CSV #########################################
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

################# GET IMAGES FOR DISPLAY #################
@app.route('/get_img')
def get_img():
    # print("get_img")
    fpath = request.args.get('fpath')
    # fpath = fpath
    list_image_name = fpath.split("/")
    # image_name = "/".join(list_image_name[-2:])
    image_name = list_image_name[-1].split('.')[0]

    if os.path.exists(fpath):
        img = cv2.imread(fpath)
    else:
        print("load 404.jph")
        img = cv2.imread("./static/images/404.jpg")

    img = cv2.resize(img, (1280, 720))

    # Tọa độ và kích thước hình chữ nhật nền
    x, y = 0, 0  # Tọa độ góc trái trên cùng của hình chữ nhật
    w, h = cv2.getTextSize(image_name, cv2.FONT_HERSHEY_SIMPLEX, 3, 6)[0]
    padding = 10  # Khoảng cách giữa văn bản và hình chữ nhật

    # Vẽ hình chữ nhật nền
    cv2.rectangle(img, (x, y), (x + w + padding, y + h + padding), (217, 217, 217), -1)  # Màu nền #D9D9D9

    # Vẽ văn bản
    cv2.putText(img, image_name, (x + padding, y + h + padding), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5, cv2.LINE_AA)  # Màu chữ đen

    ret, jpeg = cv2.imencode('.jpg', img)
    return Response((b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

########################## DOWLOAD SUBMIT FILE  #########################
@app.route('/dowload_submit_file', methods=['GET'])
def dowload_submit_file():
    print("dowload_submit_file")
    filename = request.args.get('filename')
    fpath = join("submission", filename)
    print("fpath", fpath)

    return send_file(fpath, as_attachment=True)

########################## GET FIRST ROW #########################
@app.route('/get_first_row')
def getFirstRowOfCsv():
    csv_path = "submission/submit.csv"
    result = {
        'video_id':"None",
        'frame_id':"None"
    }
    if os.path.exists(csv_path):
        lst_frame = show_csv(csv_path)[0]
        print(lst_frame)
        video_id, frame_id = lst_frame.split("/")[-2:]
        result["video_id"] = video_id
        result["frame_id"] = int(frame_id[:-4])

    return result

################# VISUALIZE FRAME SELECTED #################################
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
    pagefile_new = convertArray(pagefile)
    if query_content is not None:
        data = {'num_page': 1, 'pagefile': pagefile_new, 'query': query_content}
    else:
        data = {'num_page': 1, 'pagefile': pagefile_new}

    return render_template('index_thumb1.html', data=data)


# @app.route('/ocrfilter')
# def ocrfilter():
#     print("ocr search")

#     pagefile = []
#     text_query = request.args.get('text_ocr')

#     list_all = fill_ocr_results(text_query, ListOcrResults)
#     list_all.extend(fill_ocr_results(text_query, ListASRResults))

#     # list_all = fill_ocr_df(text_query, df_ocr)
#     # list_all = np.vstack((list_all, fill_ocr_df(text_query, df_ocr)))
    
#     print("list results of ocr + asr: ", list_all)

#     imgperindex = 100 

#     for frame in list_all:
#         list_frame_name = frame.split("/")
#         keyframe_dir = list_frame_name[0][:7]
#         video_dir = list_frame_name[0]
#         new_frame_name = list_frame_name[-1]
#         frame_in_video_path =  join("Database", "KeyFrames"+keyframe_dir, video_dir, new_frame_name)
#         frame_in_video_path =  frame_in_video_path.replace("\\","/")
#         # print("frame_in_video_path: ", frame_in_video_path)
#         if frame_in_video_path in DictKeyframe2Id:
#             print("frame_in_video_path: ", frame_in_video_path)
#             frame_id_in_video_path = DictKeyframe2Id[frame_in_video_path]
#             pagefile.append({'imgpath': frame_in_video_path, 'id': int(frame_id_in_video_path)})

#     data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    
#     return render_template('index_thumb.html', data=data)

from PIL import Image
@app.route('/img2imgs', methods=['GET', 'POST'])
def index():

    directory_path = "search_continues/uploaded/"
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        uploaded_img_path = directory_path + file.filename
        img.save(uploaded_img_path)

        idx_image, scores = search_image2image(uploaded_img_path, )
        # Lấy kết quả và gửi đến html
        id2img_fps = DictImagePath
        infos_query = list(map(id2img_fps.get, list(idx_image)))
        image_paths = [info['image_path'] for info in infos_query]
        scores = np.array(scores, dtype=np.float32).tolist()
        
        print("searching.......")
        imgperindex = 100 
        pagefile = []

        for imgpath, id, score in zip(image_paths, idx_image, scores):
            pagefile.append({'imgpath': imgpath, 'id': int(id), 'score':score})
        print("searching.........")
        
        pagefile_new = convertArray(pagefile)
        data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile_new}
    
        return render_template('index_thumb1.html', data=data)

if __name__ == '__main__':
    submit_dir = "submission"
    if not os.path.exists(submit_dir):
        os.mkdir(submit_dir)

    app.run(debug=False, host="0.0.0.0", port=5001)
