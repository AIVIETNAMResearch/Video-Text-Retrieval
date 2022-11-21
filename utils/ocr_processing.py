from rapidfuzz import fuzz

import pandas as pd 
import json
import re 
import numpy as np 

# FRAMEPATH_2_ID = json.load(open('dict\keyframe_path2id.json','r'))

# def get_top_ocr(str_query, df_ocr, k=200):  
#   df_ocr['results'] = df_ocr['text'].apply(lambda x: fuzz.token_sort_ratio(str_query, str(x)))
#   results_top= df_ocr.sort_values(by='results', ascending=False).head(k)

#   data = {}
#   for i in results_top.index:
#     row = results_top.loc[i]
#     video_id = row.video_id
#     frame_id = row.keyframe_id

#     frame_path = f'Database/KeyFrames{video_id[:-2]}/{video_id}/{frame_id:06}.jpg'
#     id_frame_path = FRAMEPATH_2_ID[frame_path]
    
#     data[frame_path] = id_frame_path

#   return data

def fill_ocr_results(st, list_ocr_results):
  def check(x):
    score = fuzz.token_set_ratio(x.lower(), st.lower())
    if score >= 80:
      if re.search(f'\s{st.lower()}\s',x.lower()):
        return True
      return False
    return False
  
  ocr_1 = list(filter(check, list_ocr_results))
  list_ocr=list(map(lambda x:f'{x.split(",")[0]}/{x.split(",")[1]:0>6}.jpg', ocr_1))
  return list_ocr

def fill_ocr_df(st, df):
  def ocr(x):
    score = fuzz.token_set_ratio(x.lower(), st.lower())
    if score > 80:
      if re.search(f'\s{st.lower()}\s',x.lower()):
        return str(score)
      else:
        return np.nan
    return np.nan
    
  df["score"] = df["ocr"].apply(ocr)
  return df.dropna(subset=["score"]).apply(lambda x: f'{x.video_id}/{x.frame_id:0>6}.jpg',axis=1).values

if __name__ == "__main__":
    with open("dict/final_info_ocr.txt", "r", encoding="utf8") as fi:
        list_ocr_results = list(map(lambda x: x.replace("\n",""), fi.readlines()))

    list_ocr = fill_ocr_results("trương mỹ hoa", list_ocr_results)
    print(list_ocr)
