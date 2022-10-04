from utils.faiss_processing import MyFaiss

bin_file='dict/faiss_cosine.bin'
json_path = 'dict/keyframes_id.json'

cosine_faiss = MyFaiss('Database', bin_file, json_path)

print(cosine_faiss.id2img_fps[0])