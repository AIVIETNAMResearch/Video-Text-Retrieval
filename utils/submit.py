import json
import os
import pandas as pd

def load_json_file(json_path: str):
    with open(json_path, 'r') as f:
        js = json.loads(f.read())

    return {int(k):v for k,v in js.items()}

def write_csv(id2img_fps, ids, des_path):
    video_names = []
    frame_ids = []
    score_ids = []

    ### GET INFOS SUBMIT ###
    # for id in ids:
    info = id2img_fps[ids] # id
    video_name = info['image_path'].split('/')[-2] + '.mp4'
    lst_frames = info['list_shot_id']

    for id_frame in lst_frames:
      video_names.append(video_name)
      frame_ids.append(id_frame)
      score_ids.append(score)
    ###########################

    ### FORMAT DATAFRAME ###
    check_files = {"video_names": video_names, "frame_ids": frame_ids, "scores": score_ids}
    df = pd.DataFrame(check_files)
    ###########################

    ### Merge csv exist file to faiss search information ###
    if os.path.exists(des_path):
      df_exist = pd.read_csv(des_path, header=None, names=["video_names", "frame_ids", "scores"])
      
      df.append(df_exist)

    ### Return DataFrame with duplicate rows removed ###
    df.drop_duplicates(subset=["video_names", "frame_ids"], inplace=True)

    ### Sort By Score ###
    df.sort_values(by=['scores'])

    ### Specifies up to 100 lines ###
    if len(df) < 99:
      df.to_csv(des_path, header=False, index=False)
      print(f"Save submit file to {des_path}")
    else:
      print('Exceed the allowed number of lines')

    return len(df), frame_ids

if __name__ == "__main__":
    ids = 1
    write_csv(json_path='./dict/keyframes_id.json', ids=ids, des_path='/content/submit.csv')