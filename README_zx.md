
# 数据
book    train1  868,49,918      test1   49
invoice train   2458,135,2594   test2   135
npx     train2  7921,441,8363   test3   441

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

python3 tools/infer/predict_system_tianchi.py \
    --det_model_dir="output/det_r50_vd_zx"  \
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

# 第一次训练只使用npx
# 第二次训练使用invoice + npx
export CUDA_VISIBLE_DEVICES='1'
nohup python3 tools/train.py \
    -c configs/det/det_r50_vd_db_zx.yml \
    -o Global.pretrain_weights=./inference/ch_ppocr_server_v2.0_det_train/ Optimizer.base_lr=0.0001 \
    >nohup.20210712_0502.train.det_r50_vd_db_zx.out &
```

# 训练识别模型

```bash
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3,4,5,6,7' tools/train.py -c configs/rec/rec_mv3_none_bilstm_ctc.yml

```


