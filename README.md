<h1>HCM AI CHALLENGE 2022 - Event Retrieval from Visual Data</h1>

---
## To do task 
- [x] CLIP4Clip
- [ ] CenterCLIP
- [x] X-CLIP
- [ ] MQVR
- [ ] TransNet
---

## Data Preparing

<b>MSRVTT</b>
```
!wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip

!wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
```

## Download Pretrain model for CLIP4Clip and X-Clip

```bash
# download CLIP（ViT-B/32） weight
wget -P ./modules https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt

# download CLIP（ViT-B/16） weight
wget -P ./modules https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

<b>Note</b>: The CLIP (ViT-B/32) is the default setting in the paper, replacing with the ViT-B/16 for better performance.

## Training

<details open>
<summary><b>CLIP4Clip</b></summary>

<details open>
<summary>Install</summary>

Clone repo and install requirements

```bash
!git clone https://github.com/ArrowLuo/CLIP4Clip.git  # clone
%cd CLIP4Clip
!pip install ftfy regex tqdm
!pip install opencv-python boto3 requests pandas
```

</details>

<details open>
<summary>Training</summary>

```bash
!python -m torch.distributed.launch --nproc_per_node=4 \
main_task_retrieval.py --do_train --num_thread_reader=0 \
--epochs=5 --batch_size=128 --n_display=50 \
--train_csv msrvtt_data/MSRVTT_train.9k.csv \
--val_csv msrvtt_data/MSRVTT_JSFUSION_test.csv \
--data_path msrvtt_data/MSRVTT_data.json \
--features_path  MSRVTT/MSRVTT_Videos \
--output_dir ckpts/ckpt_msrvtt_retrieval_looseType \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msrvtt --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/32
```
<b>Note</b>: --pretrained_clip_name can be set with ViT-B/32 or ViT-B/16

</details>

</details>

<details open>
<summary><b>X-Clip</b></summary>

<details open>
<summary>Install</summary>

Clone repo and install requirements

```bash
!git clone https://github.com/xuguohai/X-CLIP.git  # clone
%cd X-CLIP
!pip install -r requirements.txt  # install
```

</details>

<details open>
<summary>Training</summary>

<details open>
<summary>ViT-B/32</summary>

```bash
# ViT-B/32
job_name="xclip_msrvtt_vit32"
!python -m torch.distributed.launch --nproc_per_node=1 \
    main_xclip.py --do_train --num_thread_reader=2 \
    --lr 1e-4 --batch_size=16  --batch_size_val 16 \
    --epochs=10  --n_display=1 \
    --train_csv /msrvtt_data/MSRVTT_train.9k.csv \
    --val_csv /msrvtt_data/MSRVTT_JSFUSION_test.csv \
    --data_path /msrvtt_data/MSRVTT_data.json \
    --features_path /MSRVTT/videos/all \
    --output_dir ckpts_dsw/${job_name} \
    --max_words 32 --max_frames 12 \
    --datatype msrvtt --expand_msrvtt_sentences  \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32 2>&1 | tee -a log/${job_name}
```

</details>

<details open>
<summary>ViT-B/16</summary>

```bash
# ViT-B/16
job_name="xclip_msrvtt_vit16"
!python -m torch.distributed.launch --nproc_per_node=1 \
    main_xclip.py --do_train --num_thread_reader=2 \
    --lr 1e-4 --batch_size=16  --batch_size_val 16 \
    --epochs=10  --n_display=1 \
    --train_csv /msrvtt_data/MSRVTT_train.9k.csv \
    --val_csv /msrvtt_data/MSRVTT_JSFUSION_test.csv \
    --data_path /msrvtt_data/MSRVTT_data.json \
    --features_path /MSRVTT/videos/all \
    --output_dir ckpts_dsw/${job_name} \
    --max_words 32 --max_frames 12 \
    --datatype msrvtt --expand_msrvtt_sentences  \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/16 2>&1 | tee -a log/${job_name}
```

</details>

</details>

</details>