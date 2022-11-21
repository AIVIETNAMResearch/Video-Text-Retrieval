! /bin/sh
#pip3 install -r requirements.txt
# cosine bin file
gdown --id 175BDkANhtbSeWFbYxBSAMmgLq3_VpXoN
# keyframe id
gdown --id 1ZAgNnH7TLbMiyUkpKK5tbzB4RbcfyOVu
# dict_id2img_path
gdown --id 1-Hn-_nOF2ZQ9EUZrWrcNz2drEUtsFhr1
# dict_img2id_path
gdown --id 1--GJzxAtI4_UbZDfTSHlsWCiIjHpo98Q
# keyframe_path2id
gdown --id 1-341qtSpVfdHYJA8GB5nuq54uOACpvbr
# faiss bert bin
gdown --id 1-9yMp3SdS4EqGh6F8b2cBlgkJVQBFTCq
# keyframe_id_bert
gdown --id 1-FZ4PakV70_eBT9cJavFkZEaMC2hppnL
# info ocr
gdown --id 1dW_YBEtiLN2Rh1oFKQqAyOLUZoi07gTb
# info asr
gdown --id 1j3D2z6LN7fzK9SlV7u5taaWRnmjXyGpH

mv faiss_cosine.bin dict/
mv faiss_bert.bin dict/
mv dict_id2img_path.json dict/
mv dict_img2id_path.json dict/
mv keyframe_path2id.json dict/
mv keyframe_id_bert.json dict/
mv keyframe_id.json dict/
mv info_ocr dict/
mv info_asr dict/
