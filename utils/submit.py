import json
import os
import pandas as pd

def load_json_file(json_path: str):
    with open(json_path, 'r') as f:
        js = json.loads(f.read())

    return {int(k):v for k,v in js.items()}

def write_csv(id2img_fps, ids, des_path):
    des_path = os.path.join(des_path, 'submit.csv')
    check_files = []

    ### GET INFOS SUBMIT ###
    # for id in ids:
    info = id2img_fps[ids] # id
    video_name = info['image_path'].split('/')[-2]
    lst_frames = info['list_shot_id']

    for id_frame in lst_frames:
        check_files.append(os.path.join(video_name, id_frame))
    ###########################

    check_files = set(check_files)

    if os.path.exists(des_path):
        df_exist = pd.read_csv(des_path, header=None)
        lst_check_exist = df_exist.values.tolist()      
        check_exist = [info[0].replace('.mp4','/') + f"{info[1]:0>6d}" for info in lst_check_exist]

        ##### FILTER EXIST LINES FROM SUBMIT.CSV FILE #####
        check_files = [info for info in check_files if info not in check_exist]
    else:
      check_exist = []

    video_names = [i.split('/')[0] + '.mp4' for i in check_files]
    frame_ids = [i.split('/')[-1] for i in check_files]

    dct = {'video_names': video_names, 'frame_ids': frame_ids}
    df = pd.DataFrame(dct)

    if len(check_files) + len(check_exist) < 99:
      df.to_csv(des_path, mode='a', header=False, index=False)
      print(f"Save submit file to {des_path}")
    else:
      print('eExceed the allowed number of lines')

    return len(check_files) + len(check_exist), frame_ids

if __name__ == "__main__":
    ids = 1
    write_csv(json_path='./dict/keyframes_id.json', ids=ids, des_path='./')