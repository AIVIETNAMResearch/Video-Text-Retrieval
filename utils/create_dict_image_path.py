from glob import glob
import os 
import json 

def create_dict_image_path(data_dir):
    dict_json_path = {}
    for keyframe_dir in sorted(os.listdir(data_dir)):
        keyframe_dir_path = os.path.join(data_dir, keyframe_dir)
        dict_json_path[keyframe_dir[-7:]] = {}

        for subdir in sorted(os.listdir(keyframe_dir_path)):
            if subdir not in dict_json_path:
                dict_json_path[keyframe_dir[-7:]][subdir] = {}

            subdir_path = os.path.join(keyframe_dir_path, subdir)
            list_image_path = glob(os.path.join(subdir_path, "*.jpg"))
            list_image_path.sort()
            # print(list_image_path)

            for index, image_path in enumerate(list_image_path):
                image_name = image_path.split("/")[-1]
                dict_json_path[keyframe_dir[-7:]][subdir][int(index)] = image_name # id2img
                # dict_json_path[keyframe_dir[-7:]][subdir][image_name] = index # img2id 
            
            dict_json_path[keyframe_dir[-7:]][subdir]["total_image"] = len(list_image_path)

    with open('dict_image_path_id2img.json', 'w') as f:
        json.dump(dict_json_path, f)

    # with open('dict_image_path_img2id.json', 'w') as f:
    #     json.dump(dict_json_path, f)