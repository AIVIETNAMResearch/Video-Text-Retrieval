from utils.faiss_processing import MyFaiss

bin_file='dict/faiss_cosine.bin'
json_path = 'dict/keyframes_id.json'

cosine_faiss = MyFaiss('Database', bin_file, json_path)

text = 'trận bóng đá Việt Nam'
scores, _, image_paths = cosine_faiss.text_search(text, k=9)
cosine_faiss.show_images(image_paths)

print(image_paths[0])