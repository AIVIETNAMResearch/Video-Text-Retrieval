from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from craft_text_detector import Craft

import os

import cv2
import numpy as np
import pandas as pd
import time
import sys
import re
from glob import glob
from unidecode import unidecode
from collections import Counter
from tqdm import tqdm

from underthesea import text_normalize,classify
import json
import argparse
import requests
from pyvi import ViTokenizer

###########################################
URL = "https://viettelgroup.ai/nlp/api/v1/spell-checking"
headers  = {"Content-Type": "application/json",
            "token": "-i7KoYEMgUidrBn7owwHSUrdijvrHPDIeUO-PS32xdgpXMVbRkBoJlLxuGPrJG4U"}
# Spell checking            
def getAPI(input_text):
  info = {
      "sentence": input_text
  }

  resp = requests.post(URL, headers = headers, data=json.dumps(info))

  if resp.status_code != 200:
    print('error: ' + str(resp.status_code))
  else:
    content = json.loads(resp.content.decode('utf8'))
    result = content['result']
    suggestion_words = result['suggestions']
    for suggestions in suggestion_words:
      old_word = suggestions['originalText']
      new_word = suggestions['suggestion']
      input_text = input_text.replace(old_word, new_word)
  return input_text


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--json_path','--i', type=str, required=True)
  parser.add_argument('--input_csv_path', type=str)
  parser.add_argument('--output_csv_path','--o', type=str, required=True)
  parser.add_argument('--save_per_image', type=int, default=1000)
  
  args, _ = parser.parse_known_args()
  return args

def classify_text(text):
  return classify(text)

def count_letters(text):
  letter_counter = Counter(text.split(' '))
  out = {}
  for letter, count in letter_counter.items():
    out[letter] = count
  
  return json.dumps(out, ensure_ascii=False).encode('utf8')

def remove_stopwords(text):
  """
  - Tokenize the text
  - Remove stopwords
  - Return the text
  
  :param text: The text to be cleaned
  :return: A string of words that are not in the stopwords list.
  """
  text = ViTokenizer.tokenize(text)
  return " ".join([w for w in text.split() if w not in stop_words])

def format_text(text):

  # normalize with underthesea
  text = text_normalize(text)
  text = getAPI(text)
  text = remove_stopwords(text.lower())

  # format date
  res = re.findall(r"\s+([^\sA-z]+)\s?", unidecode(text))
  for x in filter(lambda x: len(x)>=6, res):
    format_x = re.sub('[^\d]', '', x)
    text = text.replace(x, format_x)
  return text

def get_ouput(video_id, frame_id,text,list_shot_id):
  res = {}
  res["video_id"] = video_id
  res["frame_id"] = frame_id
  out = count_letters(text)
  # res['collections'] = out.decode()
  # res['classify'] = classify_text(text)[0]
  res['text'] = text
  res['list_shot_id'] = list_shot_id
  return res

############################################

def main():
  
  str_json = json.load(open(args.json_path,'r'))
  csv_content = []
  
  for key,value in tqdm(str_json.items()):
    img_path = value['image_path']
    img_path = '/media/nhattuong/Data/Video-Text-Retrieval/' + img_path
    video_id = img_path.split('/')[-2]
    frame_id = img_path.split('/')[-1][:-4]
    
    if (args.input_csv_path and f'{video_id}/{frame_id}' in lst_input):
        continue
    list_shot_id = value['list_shot_id']

    frame = cv2.imread(img_path)
    if frame is None:
      break
    
    prediction_result = craft.detect_text(frame)
    boxes = prediction_result['boxes']

    # Sorted dt_boxes
    if len(boxes) > 0:
      boxes = sorted(boxes, key = lambda x:x[0][1])

    output = []
    for idx, box in enumerate(boxes):
      try:
          box = np.array(box).astype(np.int32).reshape(-1, 2)
          point1, point2, point3, point4 = box
          x, y, w, h = point1[0],point1[1],point2[0] - point1[0],point4[1]-point1[1]
          crop_img = frame[y:y+h, x:x+w]
          crop_img = Image.fromarray(crop_img)
          s = detector.predict(crop_img)
          output.append(s)
      except:
        output.append('')
    text = ' '.join(output)
    # text = format_text(' '.join(output))
    # result.append(text)
    csv_content.append( get_ouput(video_id, frame_id, text, list_shot_id))
    if len(csv_content) % args.save_per_image == 0 :
      df = pd.DataFrame(csv_content)
      df.to_csv(args.output_csv_path, index=False)
        
  craft.unload_craftnet_model()
  craft.unload_refinenet_model()  

  # Write to CSV with output_csv_path
  df = pd.DataFrame(csv_content)
  df.to_csv(args.output_csv_path, index=False)

if __name__ == "__main__":
  args = parse_args()
  config_vietorc = Cfg.load_config_from_name("vgg_transformer")
  config_vietorc['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
  config_vietorc['cnn']['pretrained']=True
  config_vietorc['device'] = 'cuda:0'
  # config_vietorc['device'] = 'cpu'
  config_vietorc['predictor']['beamsearch']=False

  if args.input_csv_path:
    csv_input = pd.read_csv(args.input_csv_path)
    lst_input = csv_input[['video_id', 'frame_id']].apply(lambda x:x.video_id+f"/{x.frame_id:06}", axis = 1).values

  detector = Predictor(config_vietorc)
  craft = Craft(output_dir=None, crop_type="box", cuda=True)
  main()