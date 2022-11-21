import numpy as np
import faiss
import glob
import json
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from utils.faiss_processing import MyFaiss

class BERTSearch(MyFaiss):
  def __init__(self, dict_bert_search='dict/keyframes_id_bert.json', bin_file='dict/faiss_bert.bin', mode='write'):
    if mode in 'search':
      self.model = SentenceTransformer('keepitreal/vietnamese-sbert')
      
      self.index = super().load_bin_file(bin_file)
      self.id2img_fps = super().load_json_file(dict_bert_search)

    else:
      pass

  def create_files(self, des_json:str, dict_support_model:str, des_bin:str, embeding_path):
    count = 0
    self.infos = []

    id2img_fps = super().load_json_file(dict_support_model)
    npy_paths = sorted(glob.glob(f'{embeding_path}/Embed*/*/*.npy'))

    index = faiss.IndexFlatL2(768)

    for npy_path in tqdm(npy_paths):
      need_path = npy_path.split('/')[-1].replace('.npy','')

      for id, values in id2img_fps.items():
        image_path = values['image_path']
        list_shot_id = values['list_shot_id']
        start, end = int(list_shot_id[0]), int(list_shot_id[-1])

        check_path = image_path.split('/')[-2] + f"_{start}_{end}"

        if need_path == check_path:
          info = {
                  "video_path": '/'.join(image_path.split('/')[:-1]),
                  "list_shot_id": list_shot_id
                }

          self.infos.append(info)
          
          try:
            feat = np.load(npy_path)
          except:
            print(npy_path)

          #### ADD FAISS ####
          feat = feat.astype(np.float32).reshape(1,-1)
          index.add(feat)  

          #### Delete ID ####
          id2img_fps.pop(id) # Delete an element from a dictionary 
          
          count += 1

          break
              
    results = dict(enumerate(self.infos))
    
    ##### SAVE JSON FILE #####
    with open(des_json, 'w') as f:
      f.write(json.dumps(results))

    ##### SAVE BIN FILE #####
    faiss.write_index(index, des_bin)
    
    ##### Print Infos Save #####
    print(f'Saved {des_json}')
    print(f"Number of Index: {count}")
    print(f'Saved {des_bin}')

  def bert_search(self, text, k):
    ###### TEXT FEATURES EXACTING ######
    text = [text, ]
    text_features = self.model.encode(text)

    ###### SEARCHING #####
    scores, idx_video = self.index.search(text_features, k=k)
    idx_video = idx_video.flatten()

    self.__infos_query = list(map(self.id2img_fps.get, list(idx_video)))
    image_paths = [os.path.join(info['video_path'],f"{info['list_shot_id'][0]}.jpg") for info in self.__infos_query]
    
    # print(f"scores: {scores}")
    # print(f"idx: {idx_video}")
    # print(f"paths: {video_path}")
    return scores, idx_video, self.__infos_query, image_paths


def main():
    # ##### CREATE FILES #####
    # create_file = BERTSearch()
    # create_file.create_files(des_json='dict/keyframes_id_bert.json', \
    #                         dict_support_model='dict/dict_support_model_batch.json', \
    #                         des_bin='dict/faiss_bert.bin',
    #                         embeding_path='/content/drive/MyDrive/ASR_Vietnamese_T')

    ##### PROCESSING #####
    mybert = BERTSearch(dict_bert_search='dict/keyframes_id_bert.json', bin_file='dict/faiss_bert.bin', mode='search')

    text = 'lũ lụt'
    scores, idx_video, infos_query, image_paths = mybert.bert_search(text, k=9)
    print(image_paths)
    # mybert.show_images(image_paths)

if __name__ == "__main__":
    main()