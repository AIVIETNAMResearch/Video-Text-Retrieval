import json
import os
import pandas as pd

def load_json_file(json_path: str):
    with open(json_path, 'r') as f:
        js = json.loads(f.read())

    return {int(k):v for k,v in js.items()}

def write_csv(json_path, ids, des_path):
    id2img_fps = load_json_file(json_path)
    des_path = os.path.join(des_path, 'submit.csv')
    check_files = []

    ### GET INFOS SUBMIT ###
    # for id in ids:
    info = id2img_fps[ids] # id
    video_name = info['image_path'].split('/')[-2]
    lst_frames = info['shot']

    for id_frame in lst_frames:
        check_files.append(os.path.join(video_name, id_frame))
    ###########################

    check_files = set(check_files)
    video_names = [i.split('/')[0] + '.mp4' for i in check_files]
    frame_ids = [i.split('/')[-1] for i in check_files]

    dct = {'video_names': video_names, 'frame_ids': frame_ids}
    df = pd.DataFrame(dct)

    df.to_csv(des_path, header=False, index=False)
    
    print(f"Save submit file to {des_path}")

if __name__ == "__main__":
    ids = 1
    write_csv(json_path='./dict/keyframes_id.json', ids=ids, des_path='./')