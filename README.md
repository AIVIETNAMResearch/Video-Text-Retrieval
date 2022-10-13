<h1>HCM AI CHALLENGE 2022 - Event Retrieval from Visual Data</h1>

---
## To do task 
- [x] [NLP_Processing](https://github.com/anminhhung/Video-Text-Retrieval/blob/main/utils/nlp_processing.py)
- [x] [Faiss_Processing](https://github.com/anminhhung/Video-Text-Retrieval/blob/main/utils/faiss_processing.py)
- [x] Reranking
- [x] [CLIP](https://github.com/openai/CLIP)
- [x] [TransNet](https://github.com/soCzech/TransNet)
- [x] OCR 
- [x] [Faster RCNN](https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1)
---
## Setup
```
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
```
## Download requirement files
- [Cosine_bin_file](https://drive.google.com/file/d/14rJ5eqEqTlDW2VxNAMr84k2TU26FB84q/view?usp=sharing).
- [Keyframes_id](https://drive.google.com/file/d/1TI6bOAV7S7xpk82uLYiHJK95HfltPJZe/view?usp=sharing).

## Format Keyframes_id.json
```
{"0": {
    "image_path": "Database/KeyFramesC00_V00/C00_V0000/000000.jpg",
    "list_shot_id": ["000000","000039","000079","000096","000118","000158"],

    "list_shot_path": [
      {"shot_id": "000000", "shot_path": "Database/KeyFramesC00_V00/C00_V0000/000000.jpg"},
      {"shot_id": "000039", "shot_path": "Database/KeyFramesC00_V00/C00_V0000/000039.jpg"},
      {"shot_id": "000079", "shot_path": "Database/KeyFramesC00_V00/C00_V0000/000079.jpg"},
      {"shot_id": "000096", "shot_path": "Database/KeyFramesC00_V00/C00_V0000/000096.jpg"},
      {"shot_id": "000118", "shot_path": "Database/KeyFramesC00_V00/C00_V0000/000118.jpg"},
      {"shot_id": "000158", "shot_path": "Database/KeyFramesC00_V00/C00_V0000/000158.jpg"}
    ]
  }
}
```

## Hướng dẫn cách sử dụng web
```
python3 web.py
```

Sau khi chạy dòng lệnh trên thì trên URL gõ đường link sau: http://0.0.0.0:5001/thumbnailimg?index=0 lúc này trang web sẽ được hiển thị lên màn hình như sau:

![ảnh UI](images/UI.png)

Ở mỗi tấm ảnh có 2 nút **knn** ở bên trái và **select** ở bên phải. Khi chúng ta ấn vào nút **knn** thì sẽ thực hiện chức năng truy vấn ảnh (tìm kiếm ảnh tương đương trong database). Lúc này sẽ xuất hiện thêm 1 tab khác show ra kết quả của truy vấn ảnh 

![KNN](images/knn.png) 

Nút **sellect** sẽ là lựa chọn ảnh đó cùng với shot ảnh của nó để ghi vào file submit 

Ở phần phía trên, nút **Search** dùng để truy vấn text, khi ta nhập câu text query và ấn nút search thì màn hình sẽ trả ra kết quả của những hình ảnh tương đương theo câu truy vấn.

Sau khi chúng ta đã xác định xong việc lựa chọn kết quả thì bấm nút **Download** để thực hiện tải file submit về để submit kết quả lên hệ thống.

Cuối cùng là nhấn nút **Clear** để reset lại file submit dưới hệ thống và tiếp thục thực hiện cho những kết quả tiếp theo