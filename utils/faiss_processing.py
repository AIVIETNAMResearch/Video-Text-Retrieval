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
from utils.nlp_processing import Translation
import clip
import torch
import pandas as pd
import re
from langdetect import detect

class File4Faiss():
  def __init__(self, root_database: str):
    self.root_database = root_database

  def write_json_file(self, json_path: str, shot_frames_path: str, option='first-end'):
    self.infos = []
    des_path = os.path.join(json_path, "keyframes_id.json")
    keyframe_paths = sorted(glob.glob(f'{self.root_database}/KeyFramesC0*_V00'))

    for kf in keyframe_paths:
      video_paths = sorted(glob.glob(f"{kf}/*"))
      for video_path in video_paths:
        image_paths = sorted(glob.glob(f'{video_path}/*.jpg'))

        ###### Get all id keyframes from video_path ######
        id_keyframes = np.array([int(id.split('/')[-1].replace('.jpg', '')) for id in image_paths])

        ###### Get scenes from video_path ######
        video_info = video_path.split('/')[-1]
        with open(f'{shot_frames_path}/{video_info}.txt', 'r') as f:
          lst_range_shotes = f.readlines()
        lst_range_shotes = np.array([re.sub('\[|\]', '', line).strip().split(' ') for line in lst_range_shotes]).astype(np.uint32)

        for im_path in image_paths:
          # im_path = '/'.join(im_path.split('/')[-3:])
          id = int(im_path.split('/')[-1].replace('.jpg', ''))

          for range_shot in lst_range_shotes:
            first, end = range_shot
            if first <= id <= end:
              break
          
          lst_shot = id_keyframes[np.where((id_keyframes>=first) & (id_keyframes<=end))]
          lst_shot = [f"{i:0>6d}" for i in lst_shot]

          info = {"image_path": im_path,
                  "shot": lst_shot}

          if option == 'full':        
            self.infos.append(info)   
          else:
            if id == first or id == end:
              self.infos.append(info)

    id2img_fps = dict(enumerate(self.infos))
    
    with open(des_path, 'w') as f:
      f.write(json.dumps(id2img_fps))

    print(f'Saved {des_path}')

  def write_bin_file(self, bin_path: str, method='L2', feature_shape=512):
    feature_paths = sorted(glob.glob(f'{self.root_database}/CLIPFeatures_C0*_V00'))

    if method in 'L2':
      index = faiss.IndexFlatL2(feature_shape)
    elif method in 'cosine':
      index = faiss.IndexFlatIP(feature_shape)
    else:
      assert f"{method} not supported"

    for feat_path in feature_paths:
      video_paths = sorted(glob.glob(f"{feat_path}/*"))
      for video in video_paths:
        feats = np.load(video)
        for feat in feats:
          feat = feat.astype(np.float32).reshape(1,-1)
          index.add(feat)  
    
    faiss.write_index(index, os.path.join(bin_path, f"faiss_{method}.bin"))

    print(f'Saved {os.path.join(bin_path, f"faiss_{method}.bin")}')

class MyFaiss():
  def __init__(self, root_database: str, bin_file: str, json_path: str):
    self.root_database = root_database

    self.index = self.load_bin_file(bin_file)
    self.id2img_fps = self.load_json_file(json_path)

    self.translater = Translation()
    
    self.__device = "cuda" if torch.cuda.is_available() else "cpu"    
    self.model, preprocess = clip.load("ViT-B/16", device=self.__device)
    
  def load_json_file(self, json_path: str):
      with open(json_path, 'r') as f:
        js = json.loads(f.read())

      return {int(k):v for k,v in js.items()}

  def load_bin_file(self, bin_file: str):
    return faiss.read_index(bin_file)

  def show_images(self, image_paths):
    fig = plt.figure(figsize=(15, 10))
    columns = int(math.sqrt(len(image_paths)))
    rows = int(np.ceil(len(image_paths)/columns))

    for i in range(1, columns*rows +1):
      img = plt.imread(image_paths[i - 1])
      ax = fig.add_subplot(rows, columns, i)
      ax.set_title('/'.join(image_paths[i - 1].split('/')[-3:]))

      plt.imshow(img)
      plt.axis("off")
      
    plt.show()

  def image_search(self, id_query, k=9):    
    query_feats = self.index.reconstruct(id_query).reshape(1,-1)

    scores, idx_image = self.index.search(query_feats, k=k)
    idx_image = idx_image.flatten()

    infos_query = list(map(self.id2img_fps.get, list(idx_image)))
    
    # image_paths = []
    # for info in infos_query:
    #   print("info: ", info)
    #   image_paths.append(info)

    image_paths = [info['image_path'] for info in infos_query if info]
    
    # print(f"scores: {scores}")
    # print(f"idx: {idx_image}")
    # print(f"paths: {image_paths}")
    
    return scores, idx_image, image_paths

  def text_search(self, text, k): #des_path_submit
    if detect(text) == 'vi':
      text = self.translater(text)

    ###### TEXT FEATURES EXACTING ######
    text = clip.tokenize([text]).to(self.__device)  
    text_features = self.model.encode_text(text).cpu().detach().numpy().astype(np.float32)

    ###### SEARCHING #####
    scores, idx_image = self.index.search(text_features, k=k)
    idx_image = idx_image.flatten()

    ###### GET INFOS KEYFRAMES_ID ######
    infos_query = list(map(self.id2img_fps.get, list(idx_image)))
    image_paths = [info['image_path'] for info in infos_query if info]
    # lst_shot = [i['shot'] for i in infos_query]
    
    ###### WRITE SUBMIT CSV FILE ######
    # self.write_csv(infos_query, k, des_path_submit)

    # print(f"scores: {scores}")
    # print(f"idx: {idx_image}")
    # print(f"paths: {image_paths}")

    return scores, idx_image, image_paths

  def write_csv(self, infos_query, k, des_path):
    des_path = os.path.join(des_path,f'submit_{k}.csv')
    check_files = []

    for info in infos_query:
      video_name = info['image_path'].split('/')[-2]
      lst_frames = info['shot']

      for id_frame in lst_frames:
        check_files.append(os.path.join(video_name, id_frame))
    
    check_files = set(check_files)
    video_names = [i.split('/')[0] + '.mp4' for i in check_files]
    frame_ids = [i.split('/')[-1] for i in check_files]

    dct = {'video_names': video_names, 'frame_ids': frame_ids}
    df = pd.DataFrame(dct)

    df.to_csv(des_path, header=False, index=False)

    print(f"Save submit file to {des_path}")

def main():
  # create_file = File4Faiss('./Database')
  # create_file.write_json_file(json_path='./')
  # create_file.write_bin_file(bin_path='./', method='cosine')

  bin_file='./faiss_cosine.bin'
  json_path = './dict/keyframes_id.json'

  cosine_faiss = MyFaiss('./Database', bin_file, json_path)

  #### Testing ####
  text = 'Người nghệ nhân đang tô màu cho chiếc mặt nạ một cách tỉ mỉ. \
        Xung quanh ông là rất nhiều những chiếc mặt nạ. \
        Người nghệ nhân đi đôi dép tổ ong rất giản dị. \
        Sau đó là hình ảnh quay cận những chiếc mặt nạ. \
        Loại mặt nạ này được gọi là mặt nạ giấy bồi Trung thu.'

  scores, _, image_paths = cosine_faiss.text_search(text, k=9, des_path_submit='./')
  cosine_faiss.show_images(image_paths)

if __name__ == "__main__":
    main()