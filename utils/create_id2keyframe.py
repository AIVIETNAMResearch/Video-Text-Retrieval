import json 

def create_id2keyframe(json_path):
    with open(json_path, 'r') as f:
        my_dict = json.loads(f.read())
    
    result_dict = {}
    for id, info in my_dict.items():
        image_path = info["image_path"]
        
        result_dict[image_path] = id 

    with open('mapping_keyframe2id.json', 'w') as f:
        json.dump(result_dict, f)