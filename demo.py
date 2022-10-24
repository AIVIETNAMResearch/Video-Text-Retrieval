from utils.process_keyframes import reformat_keyframe_name

if __name__ == "__main__":
    list_csv_paths = "dict/keyframe_p"
    list_frame_paths = "Database"

    reformat_keyframe_name(list_csv_paths, list_frame_paths)