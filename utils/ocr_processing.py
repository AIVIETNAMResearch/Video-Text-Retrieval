import rapidfuzz
from rapidfuzz import process, fuzz
import pandas as pd 

def get_top_ocr(str_query, df_ocr, k=200):  
  results = df_ocr.apply(lambda x: fuzz.token_sort_ratio(str_query, str(x['text'])), axis=1)
  df_ocr['results'] = results
  results_top= df_ocr.sort_values(by='results', ascending=False).head(k)

  return results_top

if __name__ == "__main__":
    df_ocr = pd.read_csv('dict/ocr_results.txt')
    df_ocr.columns = ["video_id ", "keyframe_id", "text"]

    path_query = 'path_query'  #/content/query-pack-0/query-1.txt
    print(get_top_ocr(path_query, df_ocr))