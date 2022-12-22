# Usage
## OCR
```
python OCR.py \
--i keyframe_id_third.json \
--o OCR_third_2.csv \
--save_per_image 100
```
**Arguments**:
- ```--i``` : File json contains image_paths
- ```--o``` : Path to save output (.csv file)
- ```--save_per_image``` : Num images to save output once

**Optional Arguments**
- ```--input_csv_path```  : Path to input csv (If you run this code before and save output to 1.csv and you wanna run continue --> You can add this argument.)

<br>
<hr>
<br>

 ## Place
```
python detect.py \
--json_path keyframes_id.json \
--output_csv_path place.csv \
--save_per_image 100
```
**Arguments**:
- ```--json_path``` : File json contains image_paths
- ```--output_csv_path``` : Path to save output (.csv file)
- ```--save_per_image``` : Num images to save output once

**Optional Arguments**
- ```--input_csv_path```  : Path to input csv (If you run this code before and save output to 1.csv and you wanna run continue --> You can add this argument.)
- ```prefix_path```  : prefix to add into image_path. Ex: Database/1.jpg => /content/Database/1.jpg if prefix_path = /content

<br>
<hr>
<br>

 ## Face
```
python face.py \
--data_dir Database \
--json_path keyframe_id_third.json \
--output_csv_path face.csv \
--save_per_image 100
```
**Arguments**:
- ```--data_dir``` : Path to images dataset
- ```--json_path``` : File json contains image_paths
- ```--output_csv_path``` : Path to save output (.csv file)
- ```--save_per_image``` : Num images to save output once

**Optional Arguments**
- ```--input_csv_path```  : Path to input csv (If you run this code before and save output to 1.csv and you wanna run continue --> You can add this argument.)

