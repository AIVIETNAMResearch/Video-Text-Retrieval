cfg = {
    'app' : {
        'host': "0.0.0.0",
        'port': 5001,
        'debug': False
    },

    # data and submit
    'submit_dir': 'submission',
    'database_dir': 'Database',

    # json dir
    'keyframe_json_path': 'dict/keyframes_id.json',
    'json_id2img_path': 'dict/dict_id2img_path.json',
    'json_img2id_path': 'dict/dict_img2id_path.json',
    'json_keyframe2id': 'dict/keyframe_path2id.json',

    # faiss
    'clip_bin_path': 'dict/faiss_cosine.bin',
    'bert_dict_path': 'dict/keyframes_id_bert.json',
    'bert_bin_path': 'dict/faiss_bert.bin',
    "clip_top_k": 200,
    "bert_top_k": 100,

    # show segment
    'around_frame': 100, # show list segment và 200 frame xung quanh frame đã chọn 

    # search image_path
    'around_path': 30, # show ảnh theo đường dẫn và 60 frame xung quanh frame đã chọn 

}