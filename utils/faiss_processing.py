# pip install faiss
# pip install ftfy regex tqdm
# pip install git+https://github.com/openai/CLIP.git
# pip install langdetect

import numpy as np
import faiss
import glob
import json
import matplotlib.pyplot as plt
import os
import math
# from utils.nlp_processing import Translation
# import clip
import torch
import pandas as pd
import re
from langdetect import detect

class File4Faiss:
  def __init__(self, root_database: str):
    self.root_database = root_database

  def re_shot_list(self, shot_list, id, k):
    len_lst = len(shot_list)
    if k>=len_lst or k == 0:
      return shot_list

    shot_list.sort()
    index_a = shot_list.index(id)

    index_get_right = k // 2
    index_get_left = k - index_get_right

    if index_a - index_get_left < 0:
      index_get_left = index_a
      index_get_right = k - index_a
    elif index_a + index_get_right >= len_lst:
      index_get_right = len_lst - index_a - 1
      index_get_left = k - index_get_right

    output = shot_list[index_a - index_get_left: index_a] + shot_list[index_a: index_a + index_get_right]
    return output

  def write_json_file(self, json_path: str, shot_frames_path: str, option='full'):
    count = 0
    self.infos = []
    des_path = os.path.join(json_path, "dict/keyframes_id.json")
    keyframe_paths = sorted(glob.glob(f'{self.root_database}/Keyframes_L*'))
    # print(keyframe_paths)
    # exit()

    for kf in keyframe_paths:
      video_paths = sorted(glob.glob(f"{kf}/*"))
      # print(video_paths)
      
      # exit()

      for video_path in video_paths:
        image_paths = sorted(glob.glob(f'{video_path}/*.jpg'))

        ###### Get all id keyframes from video_path ######
        id_keyframes = np.array([int(id.replace("\\","/").split('/')[-1].replace('.jpg', '')) for id in image_paths])
        # print(id_keyframes)
        # exit()
        
        ###### Get scenes from video_path ######
        video_path = video_path.replace('\\','/')
        video_info = video_path.split('/')[-1]
        # print(video_info)
        # exit()
        
        with open(f'{shot_frames_path}/{video_info}.txt', 'r') as f:
          lst_range_shotes = f.readlines()
        lst_range_shotes = np.array([re.sub('\[|\]', '', line).strip().split(' ') for line in lst_range_shotes]) #.astype(np.uint32)

        for im_path in image_paths:
          # print(im_path)
          # exit()
          # im_path = 'Database/' + '/'.join(im_path.split('/')[-3:])
          im_path = im_path.replace("\\","/")
          id = int(im_path.split('/')[-1].replace('.jpg', ''))
          # print(im_path)
          # print(id)
          # exit()
          
          i = 0
          flag=0
          # print(lst_range_shotes)
          # exit()
          for range_shot in lst_range_shotes:
            # print(range_shot)
            # print(type(range_shot))
            l = len(range_shot)
            i+=1
            first, end = int(range_shot[0]), int(range_shot[l-1])
            # first = int(re.sub(' ', '', first))
            # end = int(end)
            
            # print(first, end)
            # exit()

            if int(first) <= id <= int(end):
              break
            
            if i == len(lst_range_shotes):
              flag=1
          
          if flag == 1:
            print(f"Skip: {im_path}")
            print(first, end)
            continue

          ##### Get List Shot ID #####
          lst_shot = id_keyframes[np.where((id_keyframes>=first) & (id_keyframes<=end))]
          # print(lst_shot)
          lst_shot = self.re_shot_list(list(lst_shot), id, k=6)
          lst_shot = [f"{i:0>6d}" for i in lst_shot]
          # print(lst_shot)
          # exit()

          ##### Get List Shot Path #####
          lst_shot_path = []
          for id_shot in lst_shot:
            info_shot = {
                "shot_id": id_shot,
                "shot_path": '/'.join(im_path.split('/')[:-1]) + f"/{id_shot}.jpg"
            }
            lst_shot_path.append(info_shot) 

          ##### Merge All Info #####
          info = {
                  "image_path": im_path,
                  "list_shot_id": lst_shot,
                  "list_shot_path": lst_shot_path
                 }
                  
          if option == 'full':        
            self.infos.append(info)   
          else:
            if id == (end+first)//2:
              self.infos.append(info)  

          count += 1
    # exit()
    id2img_fps = dict(enumerate(self.infos))
    
    with open(des_path, 'w') as f:
      f.write(json.dumps(id2img_fps))

    print(f'Saved {des_path}')
    print(f"Number of Index: {count}")

  def load_json_(self, json_path: str):
    with open(json_path, 'r') as f:
      js = json.loads(f.read())

    return {int(k):v for k,v in js.items()}

  def write_bin_file(self, bin_path: str, json_path: str, method='L2', feature_shape=256, bin_file='./dict/faiss_blip_v1_cosine.bin'): 
    count = 0
    # print(json_path)
    id2img_fps = self.load_json_(json_path)

    # if method in 'L2':
    #   index = faiss.IndexFlatL2(feature_shape)
    # elif method in 'cosine':
    #   index = faiss.IndexFlatIP(feature_shape)
    # else:
    #   assert f"{method} not supported"
    
    index = load_bin_file(bin_file)
    
    for _, value in id2img_fps.items():
      image_path = value["image_path"]
      video_name = image_path.split('/')[-2] + '.npy'
      # print(video_name)

      # video_id = re.sub('_V\d+', '', image_path.split('/')[-2])
      video_id = image_path.split('/')[-2]
      # batch_name = image_path.split('/')[-3].split('_')[-1]
      blip_name = f"BLIP_features"
      # bert_name = './bert_obj_extract_feature'

      feat_path = os.path.join(blip_name, video_name) 
      feat_path = feat_path.replace('\\','/')
      # print(image_path)
      # exit()

      if os.path.exists(feat_path):
        feats = np.load(feat_path)
        

        ids = os.listdir(re.sub('/\d+.jpg','',image_path))
        # print(ids)
        ids = sorted(ids, key=lambda x:int(x.split('.')[0]))

        id = ids.index(image_path.split('/')[-1])
        
        print(image_path.split('/')[-2])
        print(image_path.split('/')[-1])
        print(id)
        print('------------------------------------------')
        
        # exit()
        
        feat = feats[id]
        print(feat.shape)
        feat = feat.astype(np.float32)
        # print(feat.shape)
        # exit()
        index.add(feat)
        
        count += 1
    
    faiss.write_index(index, os.path.join(bin_path, f"faiss_blip_v1_{method}_final.bin"))
    # exit()
    # faiss.write_index(index, os.path.join(bin_path, f"faiss_bert_{method}.bin"))

    print(f'Saved {os.path.join(bin_path, f"faiss_blip_{method}_final.bin")}')
    print(f"Number of Index: {count}")

def load_json_file(json_path: str):
      js = json.load(open(json_path, 'r'))

      return {int(k):v for k,v in js.items()}
    
def load_bin_file(bin_file: str):
    return faiss.read_index(bin_file)

################################################################  
## SEARCH IMAGE PATH - KEYFRAME ID   
################################################################  
def search_by_keyframeid(bin_file, id_query):
    index = load_bin_file(bin_file)
    query_feats = index.reconstruct(id_query).reshape(1,-1)
      
    scores, idx_image = index.search(query_feats, k=k)     
    idx_image = idx_image.flatten()
    scores = scores.flatten()
    return idx_image, scores
################################################################
def write_csv(infos_query, des_path):
    check_files = []
    
    ### GET INFOS SUBMIT ###
    for info in infos_query:
      video_name = info['image_path'].split('/')[-2]
      lst_frames = info['list_shot_id']

      for id_frame in lst_frames:
        check_files.append(os.path.join(video_name, id_frame))
    
    check_files = set(check_files)

    if os.path.exists(des_path):
        df_exist = pd.read_csv(des_path, header=None)
        lst_check_exist = df_exist.values.tolist()      
        check_exist = [info[0].replace('.mp4','/') + f"{info[1]:0>6d}" for info in lst_check_exist]

        ##### FILTER EXIST LINES FROM SUBMIT.CSV FILE #####
        check_files = [info for info in check_files if info not in check_exist]
    else:
      check_exist = []

    video_names = [i.split('/')[0] + '.mp4' for i in check_files]
    frame_ids = [i.split('/')[-1] for i in check_files]

    dct = {'video_names': video_names, 'frame_ids': frame_ids}
    df = pd.DataFrame(dct)

    if len(check_files) + len(check_exist) < 99:
      df.to_csv(des_path, mode='a', header=False, index=False)
      print(f"Save submit file to {des_path}")
    else:
      print('Exceed the allowed number of lines')

############# GET SUB .BIN ###################### 
def get_id2index(index_list):
  indices = []
  for index, value in enumerate(index_list):
    indices.append(index)
  return indices

def read_index_file(index_file):
  with open(index_file, 'r') as file:
      data_str = file.read()
  if ("[" in data_str and "]" in data_str):
    data_str =  data_str.replace("[", "").replace("]", "")
            
  data_list = data_str.split(", ")
  data_array = [int(num) for num in data_list]
  
  return data_array

def extract_feats_from_bin(bin_file, idx_image):
    index = faiss.read_index(bin_file)
    feats = []
    
    for idx in idx_image:
        # print(idx)
        feat = index.reconstruct(idx)
        feats.append(feat)

    feats = np.vstack(feats)
    return feats
   
def save_feats_to_bin(ids, feats, output_bin_path, output_idx_list):
    index = faiss.IndexFlatIP(256)
    index.add(feats)
    faiss.write_index(index, output_bin_path)
    arr_idx = ', '.join(str(id) for id in ids)
    # Ghi chuỗi vào tệp tin
    with open(output_idx_list, 'w') as file:
        file.write(arr_idx)
    print('done')

def mapping_index(a, b):
    mapped_array = []
    for index in b:
        mapped_array.append(a[index])
    return mapped_array

def searchcontinues(bin_file, index_file, k, id_query=None, text_features=None):
  faiss_model = load_bin_file(bin_file)
  
  data_array = read_index_file(index_file)

  if id_query is not None: # search by keyframe id
    query_feats = faiss_model.reconstruct(id_query).reshape(1,-1)
    scores, idx_images = faiss_model.search(query_feats, k=k)    
    
  elif text_features is not None: # search by text - BLIP search
    scores, idx_images = faiss_model.search(text_features, k=k)
    # print('--------------------------------------------------------------')
    
    # print(idx_images) ## --> trả về vị trí (index)

  scores = scores.flatten()
  idx_images = idx_images.flatten()
  idx_images = mapping_index(data_array, idx_images) 
  # print('================================================================')
  # print(idx_images)  ## --> trả về giá trị (value)
  
  return scores, idx_images

# Hàm để lấy tất cả các ID  
def get_all_ids(data):
    ids = []
    for item in data:
        if 'list_frame' in item and isinstance(item['list_frame'], list):
            for frame in item['list_frame']:
                if 'id' in frame:
                    ids.append(frame['id'])

    return ids
  
def save_bin_delete_noise(binfile, index_file, new_bin_path):
  index_list = read_index_file(index_file)
  ids = get_id2index(index_list)
  
  # for i in range(0, len(ids)):
  #   print('index_list: ',index_list[i])
  #   print('ids: ',ids[i])
  
  feats = extract_feats_from_bin(binfile, ids)

  # savefile sub bin and idx of frames
  index = faiss.IndexFlatIP(256)
  index.add(feats)
  faiss.write_index(index, new_bin_path)
  
############### REMOVE VIDEO ID NOISE ################  
import json

def remove_keys_and_save(input_video_id, json_file_path):
    # Đọc dữ liệu từ tệp JSON ban đầu
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    
    # Tạo một bản sao của dữ liệu JSON để tránh ảnh hưởng đến dữ liệu gốc
    new_data = dict(data)
    
    # Xóa các key chứa video id đầu vào
    for key in list(new_data.keys()):
        if input_video_id in key:
            del new_data[key]
    
    # print(new_data)
    # Lưu dữ liệu mới vào tệp txt
    values = [int(value) for value in list(new_data.values())]
    return values, new_data

###############  SEARCH TAGS  ################################
# lấy những row có giá trị trong cột obj >= số lượng nhập vào (ví dụ: 1 man, 2 woman --> lấy ra những thằng
#     có cột man >= 1 và cột woman >=2)
def search_tags(csv_filename, obj_query):
    df = pd.read_csv(csv_filename)
    # Chuyển input và tên cột thành chữ thường
    obj_query = obj_query.lower()
    df.columns = df.columns.str.lower()
    input_columns = []

    # Tạo một mask (đánh dấu) cho các dòng thỏa mãn điều kiện
    conditions = []
    for condition in obj_query.split(','):
        value, column = condition.strip().split(' ')
        column = column.strip()
        value = float(value.strip())
        input_columns.append(column)
        
        conditions.append(df[column] >= value)
    mask = pd.concat(conditions, axis=1).all(axis=1)

    # Lấy ra các dòng thỏa mãn điều kiện và chỉ các cột cần thiết
    # columns_to_select = ['index', 'img_paths'] + input_columns
    # df_new = df.loc[mask, columns_to_select]
    df_new = df.loc[mask]
    list_index = df_new['index']
    img_paths = df_new['img_paths']

    return df_new, list_index, img_paths



########################################################################################## 
###               SEARCH IMAGE 2 IMAGES
############################################################################################
import sys
import os
import torch
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))

# Xác định đường dẫn tới thư mục LAVIS
lavis_dir = os.path.join(current_dir, 'LAVIS')

# Thêm đường dẫn tương đối của thư mục LAVIS vào sys.path
sys.path.append(lavis_dir)
from lavis.models import load_model_and_preprocess

def search_image2image(img_path, bin_file, k):
  raw_image = Image.open(img_path).convert("RGB")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)
  image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
  sample = {"image": image}
  query_feats = model.extract_features(sample, mode="image").reshape(1,-1)
  
  index = load_bin_file(bin_file)
  
  scores, idx_image = index.search(query_feats, k=k)     
  idx_image = idx_image.flatten()
  scores = scores.flatten()
  
  return idx_image, scores


# create_file = File4Faiss('Database')
# create_file.write_json_file(json_path='./', shot_frames_path='./scenes_txt')
# create_file.write_bin_file(bin_path='./dict/', json_path='./dict/dict_final/keyframes_id.json', method='cosine', feature_shape=256) # Bert model
# def main():
  
  ### CREATE JSON AND BIN FILES #####
  

  # ##### TESTING #####
  # bin_file='dict/faiss_cosine.bin'
  # json_path = '/dict/keyframes_id.json'

  # cosine_faiss = MyFaiss('./Database', bin_file, json_path)

  # ##### IMAGE SEARCH #####
  # i_scores, _, infos_query, i_image_paths = cosine_faiss.image_search(id_query=9999, k=9)
  # # cosine_faiss.write_csv(infos_query, des_path='/content/submit.csv')
  # cosine_faiss.show_images(i_image_paths)

  # ##### TEXT SEARCH #####
  # text = 'Người nghệ nhân đang tô màu cho chiếc mặt nạ một cách tỉ mỉ. \
  #       Xung quanh ông là rất nhiều những chiếc mặt nạ. \
  #       Người nghệ nhân đi đôi dép tổ ong rất giản dị. \
  #       Sau đó là hình ảnh quay cận những chiếc mặt nạ. \
  #       Loại mặt nạ này được gọi là mặt nạ giấy bồi Trung thu.'

  # scores, _, infos_query, image_paths = cosine_faiss.text_search(text, k=9)
  # # cosine_faiss.write_csv(infos_query, des_path='/content/submit.csv')
  # cosine_faiss.show_images(image_paths)
  
  
