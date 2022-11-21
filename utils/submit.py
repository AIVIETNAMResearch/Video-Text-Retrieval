import json
import os
import pandas as pd
import copy

def load_json_file(json_path: str):
    with open(json_path, 'r') as f:
        js = json.loads(f.read())

    return {int(k):v for k,v in js.items()}

def write_csv(id2img_fps, mode_write_csv, selected_image, id, des_path):
    des_path = os.path.join(des_path, 'submit.csv')
    video_names = []
    frame_ids = []

    ### GET SELECTED SUBMIT ###
    video_name_selected = selected_image.split('/')[-2] + '.mp4'
    id_frame_selected = selected_image.split('/')[-1].replace('.jpg', '')

    video_names.append(video_name_selected)
    frame_ids.append(id_frame_selected)
    ############################

    if mode_write_csv == 'list_shot':
      ### GET INFOS SUBMIT ###
      info = copy.deepcopy(id2img_fps[id])
      video_name = info['image_path'].split('/')[-2] + '.mp4'
      lst_frames = info['list_shot_id']
      if id_frame_selected in lst_frames:
        lst_frames.remove(id_frame_selected)

      for id_frame in lst_frames:
        video_names.append(video_name)
        frame_ids.append(id_frame)
      #########################

    ### FORMAT DATAFRAME ###
    check_files = {"video_names": video_names, "frame_ids": [int(i) for i in frame_ids]}
    df_selected = pd.DataFrame(check_files)
    ###########################

    if os.path.exists(des_path):
        df_exist = pd.read_csv(des_path, names=["video_names", "frame_ids"])
        ##### Find Rows in df_selected Which Are Not Available in df_exist #####
        df_save = df_selected.merge(df_exist, how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', 1)
    else:
      df_exist = {}
      df_save = df_selected

    if len(df_save) + len(df_exist) < 99:
      df_save.to_csv(des_path, mode='a', header=False, index=False)
      print(f"Save submit file to {des_path}")
    else:
      print('Exceed the allowed number of lines')

    return len(df_save) + len(df_exist), frame_ids

def show_csv(csv_path):
  submit_csv = pd.read_csv(csv_path, header = None)

  # x[0]: C00_V0135.mp4
  # x[1]: 013053
  # f'Database/KeyFrames{str(x[0])[:-6]}/{x[0][:-4]}/{x[1]:06}.jpg' => Frame Path
  submit_csv['path'] = submit_csv.apply(lambda x: f'Database/KeyFrames{str(x[0])[:-6]}/{x[0][:-4]}/{x[1]:06}.jpg', axis = 1)
  paths = submit_csv['path'].values
  return paths

if __name__ == "__main__":
    ids = 1
    write_csv(json_path='./dict/keyframes_id.json', ids=ids, des_path='./')