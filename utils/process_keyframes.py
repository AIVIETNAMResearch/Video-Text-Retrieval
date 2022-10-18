import os
import cv2 
import glob

def resize_keyframes(Database_path):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    img_paths = glob.glob(f'{Database_path}/KeyFrames*/*/*.jpg')

    for img_path in img_paths:
        print("img_path: ", img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224,224))

        os.system(f'rm {img_path}')
        cv2.imwrite(img_path, img)