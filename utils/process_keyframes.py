import os
import cv2 
from glob import glob
import pandas as pd
from tqdm import tqdm

def resize_keyframes(Database_path):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    img_paths = glob.glob(f'{Database_path}/KeyFrames*/*/*.jpg')

    for img_path in img_paths:
        print("img_path: ", img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224,224))

        os.system(f'rm {img_path}')
        cv2.imwrite(img_path, img)
        
def reformat_keyframe_name(list_csv_paths:str, list_frame_paths:str):
    """
    It takes a list of csv files and a list of frame paths, and renames the frames in the frame paths to
    match the csv files
    
    :param list_csv_paths: the path to the folder containing the csv files. If folder contains Batch1 and Batch2 csv then function will rename all frame in Batch1 and Batch2.
    :param list_frame_paths: the path to the folder containing the frames
    """
    lst_csv = glob(f'{list_csv_paths}/*.csv')
    print("lst_csv: ", lst_csv)
    lst_csv.sort()
    dct_names = {}

    for csv_path in tqdm(lst_csv):
        df = pd.read_csv(csv_path,header = None)
        for i in df.index:
            row = df.loc[i]
            video_id = csv_path.split('/')[-1][:-4]
            key = f'{video_id}/{row[0]}'
            value = f'{video_id}/{row[1]:06}.jpg' 
            dct_names[key] = value

    prev_keyframe = ''
    for key, value in tqdm(dct_names.items()):
        keyframe = f'KeyFrames{key.split("/")[0][:-2]}' # KeyFramesC00_V00
        frame_src_path = f'{list_frame_paths}/{keyframe}/{keyframe}/{key}'
        frame_dst_path = f'{list_frame_paths}/{keyframe}/{keyframe}/{value}'

        if frame_src_path == frame_dst_path or not os.path.exists(frame_src_path):
            continue

        if prev_keyframe != keyframe:
            lst_frame_in_video = os.listdir('/'.join(frame_src_path.split('/')[:-1]))
            prev_keyframe = keyframe

        if frame_dst_path.split('/')[-1] in     lst_frame_in_video:
            os.remove(frame_src_path)
        else:
            os.rename(frame_src_path, frame_dst_path)