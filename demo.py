from utils.ocr_processing import get_top_ocr
import json 
import pandas as pd 

if __name__ == "__main__":
    df_ocr = pd.read_csv('dict/ocr_results.txt')
    df_ocr.columns = ["video_id ", "keyframe_id", "text"]

    path_query = 'Thẩm định tượng Đức Thánh Trần ở Hồ Mây, Vũng Tàu'

    df_top = get_top_ocr(path_query, df_ocr, 50)
    # for index, row in df_top.iterrows():
    #     print(row)
    #     break

    df_top.to_csv("demo.csv")