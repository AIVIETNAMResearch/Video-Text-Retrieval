# pip install faiss
# pip install ftfy regex tqdm
# pip install git+https://github.com/openai/CLIP.git

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

class File4Faiss():
  def __init__(self, root_database: str):
    self.root_database = root_database

  def write_json_file(self, json_path: str):
    des_path = os.path.join(json_path, "keyframes_id.json")
    files = []
    keyframe_paths = sorted(glob.glob(f'{self.root_database}/KeyFramesC0*_V00'))

    for kf in keyframe_paths:
      video_paths = sorted(glob.glob(f"{kf}/*"))
      for video_path in video_paths:
        image_paths = sorted(glob.glob(f'{video_path}/*.jpg'))
        for im_path in image_paths:
          # im_path = '/'.join(im_path.split('/')[-3:])
          files.append(im_path)     
    id2img_fps = dict(enumerate(files))
    
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

  def __call__(self, id_query, k=9):    
    query_feats = self.index.reconstruct(id_query).reshape(1,-1)
    scores, idx_image = self.index.search(query_feats, k=k)
    idx_image = idx_image.flatten()
    image_paths = list(map(self.id2img_fps.get, list(idx_image)))
    
    # print(f"scores: {scores}")
    # print(f"idx: {idx_image}")
    # print(f"paths: {image_paths}")
    return scores, idx_image, image_paths

  def predict(self, text, k=9):
    text = self.translater(text)
    print(f'Translation: {text}')
    text = clip.tokenize([text]).to(self.__device)  
    text_features = self.model.encode_text(text).cpu().detach().numpy().astype(np.float32)

    scores, idx_image = self.index.search(text_features, k=k)
    idx_image = idx_image.flatten()
    image_paths = list(map(self.id2img_fps.get, list(idx_image)))
    
    # print(f"scores: {scores}")
    # print(f"idx: {idx_image}")
    # print(f"paths: {image_paths}")
    return scores, idx_image, image_paths

def main():
  # create_file = File4Faiss('./Database')
  # create_file.write_json_file(json_path='./')
  # create_file.write_bin_file(bin_path='./', method='cosine')

  bin_file='./faiss_cosine.bin'
  json_path = './keyframes_id.json'
  cosine_faiss = MyFaiss('./Database', bin_file, json_path)

  #### Testing ####
  text = 'trận bóng đá Việt Nam'
  scores, _, image_paths = cosine_faiss.predict(text, k=9)
  cosine_faiss.show_images(image_paths)

if __name__ == "__main__":
    main()