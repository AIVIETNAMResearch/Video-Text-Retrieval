
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import argparse
from glob import glob
from random import shuffle
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import json
print(cv2.__version__)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--json_path', type=str, help ="Path to json file contain image path")
  parser.add_argument('--prefix_path', type=str, default='Database', help="prefix to add into image_path. Ex: Database/1.jpg => /content/Database/1.jpg if prefix_path=/content")
  parser.add_argument('--input_csv_path', '--i', type=str, help="File csv that you want to add to output if you have run before")
  parser.add_argument('--output_csv_path','--o', type=str, required=True, help = "Path to output csv")
  parser.add_argument('--save_per_image', type=int, default=1000, help="Num image to save output")
  
  args, _ = parser.parse_known_args()

  
  return args

if __name__ == "__main__":

  args = parse_args()
  top_k = 3

  dct = json.load(open(args.json_path, 'r'))
  if args.input_csv_path:
    df_input = pd.read_csv(args.input_csv_path)
    lst_video_frame = df_input[["video_id","frame_id"]].apply(lambda x: f"{x.video_id}/{x.frame_id}", axis=1).values

  with open('places_id2vi_id.txt', 'r')as fi:
    lines = list(map(lambda x: x.replace('\n',''), fi.readlines()))
  st = ''.join(lines)
  en2vi = json.loads(st.replace("'",'"'))

  


  prototxtFilePath = 'deploy_vgg16_places365.prototxt'
  if not os.path.exists(prototxtFilePath):
    os.system(f'gdown --id  10bLEiaDnWlRXMr3_ZuCGxe_yjpcr4NCB')

  modelFilePath = 'vgg16_places365.caffemodel'
  if not os.path.exists(modelFilePath):
    os.system(f'wget http://places2.csail.mit.edu/models_places365/vgg16_places365.caffemodel')
  # prototxtFilePath = '/content/drive/MyDrive/AIC_HCM/PlaceClassify/places365/deploy_vgg16_places365.prototxt'
  # modelFilePath = '/content/drive/MyDrive/AIC_HCM/PlaceClassify/weights/vgg16_places365.caffemodel'
  
  net = cv2.dnn.readNetFromCaffe(prototxtFilePath,modelFilePath)

  meanValues=(104, 127, 123)
  # meanValues=(124, 127, 123)
  imgWidth=224
  imgHeight=224

  # load the image transformer
  centre_crop = trn.Compose([
          trn.Resize((256,256)),
          trn.CenterCrop(224),
          trn.ToTensor(),
          trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])

  # load the class label
  file_name = 'categories_places365.txt'
  if not os.access(file_name, os.W_OK):
      synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
      os.system('wget ' + synset_url)
  classes = list()
  with open(file_name) as class_file:
      for line in class_file:
          classes.append(line.strip().split(' ')[0][3:])
  classes = tuple(classes)

  csv_content = []
  count = 0
  for key, value in tqdm(dct.items()):
    img_path = value["image_path"]
    split_image_path = img_path.split("/")
    img_path = f'{args.prefix_path}/{"/".join(split_image_path[1:])}'
    
    video_frame = '/'.join(split_image_path[-2:])
    if (args.input_csv_path and video_frame in lst_video_frame) or not os.path.exists(img_path):
      continue
    img = cv2.imread(img_path)
    count+=1
    
    video_id = split_image_path[-2]
    frame_id = split_image_path[-1]
    list_shot_id = value["list_shot_id"]


    blob = cv2.dnn.blobFromImage(img, 1, (imgWidth, imgHeight), meanValues)
    net.setInput(blob)
    # This method returns a list of probabilities of all cases , it is of shape ((1, Number of classes)
    preds = net.forward()[0]
    idx_max =  preds.argsort()[-top_k:][::-1]

    pred_output = {}
    for idx in idx_max:
      pred = classes[idx]
      percent = round(preds[idx] * 100, 3)
      pred_output[en2vi[re.sub('[_/]',' ',pred)].lower()] = f'{percent}%'

    csv_content.append({
      'video_id': video_id,
      'frame_id': frame_id,
      'collections': pred_output,
      'list_shot_id':list_shot_id
    })

    if count % args.save_per_image == 0 :
      df = pd.DataFrame(csv_content)
      df.to_csv(args.output_csv_path, index = False)

  df = pd.DataFrame(csv_content)
  df.to_csv(args.output_csv_path, index = False)