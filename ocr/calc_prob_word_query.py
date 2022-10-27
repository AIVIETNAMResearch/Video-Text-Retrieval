import rapidfuzz
from rapidfuzz import process, fuzz

ocr_final = pd.read_csv('/content/drive/MyDrive/HCM_AI_Challenge_2022/database/ocr_final.txt')
ocr_final.columns = ["video_id ", "keyframe_id", "text"]

def top200(path_query, ocr):
  with open(path_query) as f:
    query = f.read()
  
  results = ocr.apply(lambda x: fuzz.token_sort_ratio(query, str(x['text'])), axis=1)
  ocr['results'] = results
  results_top200 = ocr.sort_values(by='results', ascending=False).head(200)

  return results_top200

path_query = 'path_query'  #/content/query-pack-0/query-1.txt
print(top200(path_query, ocr_final))
