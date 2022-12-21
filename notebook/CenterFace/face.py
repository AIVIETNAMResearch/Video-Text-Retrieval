import os
import argparse
import json
import pandas as pd

import numpy as np
import cv2
from tqdm import tqdm

class CenterFace(object):
  def __init__(self, landmarks=True):
    self.landmarks = landmarks
    if not os.path.exists("centerface.onnx"):
      os.system("gdown --id 13zEcqVw5VsWob5kZ5neRSdn60smiDSRW")
    if self.landmarks:
      self.net = cv2.dnn.readNetFromONNX('centerface.onnx')
    else:
      self.net = cv2.dnn.readNetFromONNX('cface.1k.onnx')
    self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = 0, 0, 0, 0

  def __call__(self, img, height, width, threshold=0.5):
    self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = self.transform(height, width)
    return self.inference_opencv(img, threshold)

  def inference_opencv(self, img, threshold):
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(self.img_w_new, self.img_h_new), mean=(0, 0, 0), swapRB=True, crop=False)
    self.net.setInput(blob)
    if self.landmarks:
      heatmap, scale, offset, lms = self.net.forward(["537", "538", "539", '540'])
    else:
      heatmap, scale, offset = self.net.forward(["535", "536", "537"])
    return self.postprocess(heatmap, lms, offset, scale, threshold)

  def transform(self, h, w):
    img_h_new, img_w_new = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
    scale_h, scale_w = img_h_new / h, img_w_new / w
    return img_h_new, img_w_new, scale_h, scale_w

  def postprocess(self, heatmap, lms, offset, scale, threshold):
    if self.landmarks:
      dets, lms = self.decode(heatmap, scale, offset, lms, (self.img_h_new, self.img_w_new), threshold=threshold)
    else:
      dets = self.decode(heatmap, scale, offset, None, (self.img_h_new, self.img_w_new), threshold=threshold)
    if len(dets) > 0:
      dets[:, 0:4:2], dets[:, 1:4:2] = dets[:, 0:4:2] / self.scale_w, dets[:, 1:4:2] / self.scale_h
      if self.landmarks:
        lms[:, 0:10:2], lms[:, 1:10:2] = lms[:, 0:10:2] / self.scale_w, lms[:, 1:10:2] / self.scale_h
    else:
      dets = np.empty(shape=[0, 5], dtype=np.float32)
      if self.landmarks:
        lms = np.empty(shape=[0, 10], dtype=np.float32)
    if self.landmarks:
      return dets, lms
    else:
      return dets

  def decode(self, heatmap, scale, offset, landmark, size, threshold=0.1):
    heatmap = np.squeeze(heatmap)
    scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
    offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
    c0, c1 = np.where(heatmap > threshold)
    if self.landmarks:
      boxes, lms = [], []
    else:
      boxes = []
    if len(c0) > 0:
      for i in range(len(c0)):
        s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
        o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
        s = heatmap[c0[i], c1[i]]
        x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(0, (c0[i] + o0 + 0.5) * 4 - s0 / 2)
        x1, y1 = min(x1, size[1]), min(y1, size[0])
        boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])
        if self.landmarks:
          lm = []
          for j in range(5):
            lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
            lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
          lms.append(lm)
      boxes = np.asarray(boxes, dtype=np.float32)
      keep = self.nms(boxes[:, :4], boxes[:, 4], 0.3)
      boxes = boxes[keep, :]
      if self.landmarks:
        lms = np.asarray(lms, dtype=np.float32)
        lms = lms[keep, :]
    if self.landmarks:
      return boxes, lms
    else:
      return boxes

  def nms(self, boxes, scores, nms_thresh):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores)[::-1]
    num_detections = boxes.shape[0]
    suppressed = np.zeros((num_detections,), dtype=bool)

    keep = []
    for _i in range(num_detections):
      i = order[_i]
      if suppressed[i]:
        continue
      keep.append(i)

      ix1 = x1[i]
      iy1 = y1[i]
      ix2 = x2[i]
      iy2 = y2[i]
      iarea = areas[i]

      for _j in range(_i + 1, num_detections):
        j = order[_j]
        if suppressed[j]:
          continue

        xx1 = max(ix1, x1[j])
        yy1 = max(iy1, y1[j])
        xx2 = min(ix2, x2[j])
        yy2 = min(iy2, y2[j])
        w = max(0, xx2 - xx1 + 1)
        h = max(0, yy2 - yy1 + 1)

        inter = w * h
        ovr = inter / (iarea + areas[j] - inter)
        if ovr >= nms_thresh:
          suppressed[j] = True

    return keep


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str)
  parser.add_argument('--json_path', type=str)
  parser.add_argument('--input_csv_path', '--i', type=str)
  parser.add_argument('--output_csv_path','--o', type=str, required=True)
  parser.add_argument('--save_per_image', type=int, default=1000)
  
  args, _ = parser.parse_known_args()
  return args


def test_image_tensorrt(image_path):
  frame = cv2.imread(image_path)
  H, W = frame.shape[:2]
  dets, lms = centerface(frame, H, W, threshold=0.35)

  out = []
  for det in dets:
    boxes, score = det[:4], det[4]
    x,y,x2,y2 = int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3])
    out.append({
      "prob": score,
      "ratio": round(((x2-x)*(y2-y))/(W*H), 5),
      "bbox": [x, y, x2, y2]
    })
  return out



def main():
  dct = json.load(open(args.json_path, 'r'))
  if args.input_csv_path:
    df_input = pd.read_csv(args.input_csv_path)
    lst_video_frame = df_input[["video_id","frame_id"]].apply(lambda x: f"{x.video_id}/{x.frame_id:06}.jpg", axis=1).values
  
  csv_content = []
  count = 0
  for key, value in tqdm(dct.items()):
    image_path = value["image_path"]
    split_image_path = image_path.split("/")
    list_shot_id = value["list_shot_id"]
    video_frame = '/'.join(split_image_path)[-2:]
    
    if args.input_csv_path and video_frame in lst_video_frame:
      continue

    video_id = split_image_path[-2]
    frame_id = split_image_path[-1]

    image_path = '/media/nhattuong/Data/Video-Text-Retrieval/' + image_path
    out = test_image_tensorrt(image_path)
    csv_content.append({
      'video_id': video_id,
      'frame_id': frame_id,
      'collections': out,
      'list_shot_id':list_shot_id
    })
  
    if count % args.save_per_image == 0:
      df = pd.DataFrame(csv_content)
      df.to_csv(args.output_csv_path, index = False)
  df = pd.DataFrame(csv_content)
  df.to_csv(args.output_csv_path, index = False)
  
if __name__ == '__main__':
  args = parse_args()
  centerface = CenterFace(landmarks=True)
  main()
