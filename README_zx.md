
# 数据
book    train1  868,49,918      test1   49
invoice train   2458,135,2594   test2   135
npx     train2  7921,441,8363   test3   441

```
     868 data/det_data/Xeon1OCR_round1_train1_20210526.csv.train.txt
      49 data/det_data/Xeon1OCR_round1_train1_20210526.csv.val.txt
    2458 data/det_data/Xeon1OCR_round1_train_20210524.csv.train.txt
     135 data/det_data/Xeon1OCR_round1_train_20210524.csv.val.txt
    7921 data/det_data/Xeon1OCR_round1_train2_20210526.csv.train.txt
     441 data/det_data/Xeon1OCR_round1_train2_20210526.csv.val.txt

      50 data/labels/Xeon1OCR_round1_test1_20210528.csv
     136 data/labels/Xeon1OCR_round1_test2_20210528.csv
     442 data/labels/Xeon1OCR_round1_test3_20210528.csv
     918 data/labels/Xeon1OCR_round1_train1_20210526.csv
    2594 data/labels/Xeon1OCR_round1_train_20210524.csv
    8363 data/labels/Xeon1OCR_round1_train2_20210526.csv
```


# 生成结果
```bash
python3 tools/infer/predict_system_tianchi.py \
    --det_model_dir="./inference/ch_ppocr_server_v2.0_det_infer/"  \
    --rec_model_dir="./inference/ch_ppocr_server_v2.0_rec_infer/" \
    --cls_model_dir='./inference/ch_ppocr_mobile_v2.0_cls_infer/' \
    --use_angle_cls=True \
    --use_space_char=True

zip -r submit.zip Xeon1OCR_round1_test*


# 模型转换
python3 tools/export_model.py \
    -c configs/det/det_r50_vd_db_zx.yml \
    -o Global.pretrained_model=output/det_r50_vd_zx/best_accuracy  Global.save_inference_dir=output/det_r50_vd_zx/

# test1 book
python3 tools/infer/predict_system_tianchi.py \
    --det_model_dir="./inference/ch_ppocr_server_v2.0_det_infer"  \
    --rec_model_dir="./inference/ch_ppocr_server_v2.0_rec_infer/" \
    --cls_model_dir='./inference/ch_ppocr_mobile_v2.0_cls_infer/' \
    --use_angle_cls=True \
    --use_space_char=True

# test2 invoice
python3 tools/infer/predict_system_tianchi.py \
    --det_model_dir="./output/det_r50_vd_zx.invoice"  \
    --rec_model_dir="./inference/ch_ppocr_server_v2.0_rec_infer/" \
    --cls_model_dir='./inference/ch_ppocr_mobile_v2.0_cls_infer/' \
    --use_angle_cls=True \
    --use_space_char=True
# test npx
python3 tools/infer/predict_system_tianchi.py \
    --det_model_dir="./output/det_r50_vd_zx.npx"  \
    --rec_model_dir="./inference/ch_ppocr_server_v2.0_rec_infer/" \
    --cls_model_dir='./inference/ch_ppocr_mobile_v2.0_cls_infer/' \
    --use_angle_cls=True \
    --use_space_char=True

```


# 训练检测模型
```bash

python3 tools/train.py -c configs/det/det_mv3_db.yml -o Optimizer.base_lr=0.0001
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/det/det_mv3_db.yml -o Optimizer.base_lr=0.0001



python3 tools/train.py \
    -c configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml \
    -o Global.pretrain_weights=./inference/ch_ppocr_server_v2.0_det_train/

# 第一次训练只使用invoice
export CUDA_VISIBLE_DEVICES='1'
nohup python3 tools/train.py \
    -c configs/det/det_r50_vd_db_zx.invoice.yml \
    -o Global.pretrain_weights=./inference/ch_ppocr_server_v2.0_det_train/ \
    >nohup.20210712_0650.train.det_r50_vd_db_zx.invoice.out &
step    28000   hmean   0.6451
# 第二次训练使用invoice + npx
export CUDA_VISIBLE_DEVICES='1'
nohup python3 tools/train.py \
    -c configs/det/det_r50_vd_db_zx.yml \
    -o Global.pretrain_weights=./inference/ch_ppocr_server_v2.0_det_train/ Optimizer.base_lr=0.0001 \
    >nohup.20210712_0650.train.det_r50_vd_db_zx.out &
step    50000   hmean   0.57062e
# 第三次训练npx
export CUDA_VISIBLE_DEVICES='0'
nohup python3 tools/train.py \
    -c configs/det/det_r50_vd_db_zx.npx.yml \
    -o Global.pretrain_weights=./inference/ch_ppocr_server_v2.0_det_train/ Optimizer.base_lr=0.0001 \
    >nohup.20210712_1850.train2.det_r50_vd_db_zx.npx.out &
step    54000   hmean   0.58004
```

# 训练识别模型

```bash
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3,4,5,6,7' tools/train.py -c configs/rec/rec_mv3_none_bilstm_ctc.yml

```

# 验证

```bash
python3 tools/eval.py \
    -c configs/det/det_r50_vd_db_zx.invoice.yml \
    -o Global.checkpoints=./pretrain_models/ch_ppocr_server_v2.0_det_train/best_accuracy Global.use_gpu=false


python3 tools/eval.py \
    -c configs/det/det_r50_vd_db_zx.npx.yml \
    -o Global.pretrain_weights=./inference/ch_ppocr_server_v2.0_det_train/

```
