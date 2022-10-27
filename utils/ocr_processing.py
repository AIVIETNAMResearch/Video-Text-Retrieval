import rapidfuzz
from rapidfuzz import process, fuzz
import pandas as pd 
import json

FRAMEPATH_2_ID = json.load(open('dict\keyframe_path2id.json','r'))

def get_top_ocr(str_query, df_ocr, k=200):  
  df_ocr['results'] = df_ocr['text'].apply(lambda x: fuzz.token_sort_ratio(str_query, str(x)))
  results_top= df_ocr.sort_values(by='results', ascending=False).head(k)

  data = {}
  for i in results_top.index:
    row = results_top.loc[i]
    video_id = row.video_id
    frame_id = row.keyframe_id

    frame_path = f'Database/KeyFrames{video_id[:-2]}/{video_id}/{frame_id:06}.jpg'
    id_frame_path = FRAMEPATH_2_ID[frame_path]
    
    data[frame_path] = id_frame_path

  return data

if __name__ == "__main__":
    df_ocr = pd.read_csv('dict/ocr_results.txt')
    df_ocr.columns = ["video_id", "keyframe_id", "text"]
    df_ocr.drop_duplicates(inplace=True)

    path_query = 'path_query'  #/content/query-pack-0/query-1.txt
    print(get_top_ocr(path_query, df_ocr))