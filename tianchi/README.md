

# 1 下载图片
```bash
# invoice
python3 down_image.py \
    --input ../input/复赛/OCR复赛数据集01.csv \
    --out_dir   ../data/imgs1
# npx
python3 down_image.py \
    --input ../input/复赛/OCR复赛数据集02.csv \
    --out_dir   ../data/imgs2
```

# 2 读取数据并拆分训练集／验证集
python3 split_train_val.py \
    --input /media/zhangxin/data1/data_public/tianchi/复赛/OCR复赛数据集01.csv \
    --imgs_dir /media/zhangxin/data1/data_public/tianchi/复赛/imgs1 \
    --train_dir /media/zhangxin/data1/data_public/tianchi/复赛/imgs1_train \
    --val_dir /media/zhangxin/data1/data_public/tianchi/复赛/imgs1_val \
    --train_file /media/zhangxin/data1/data_public/tianchi/复赛/train1.txt \
    --val_file /media/zhangxin/data1/data_public/tianchi/复赛/val1.txt \
    --val_num 400


python3 split_train_val.py \
    --input /media/zhangxin/data1/data_public/tianchi/复赛/OCR复赛数据集02.csv \
    --imgs_dir /media/zhangxin/data1/data_public/tianchi/复赛/imgs2 \
    --train_dir /media/zhangxin/data1/data_public/tianchi/复赛/imgs2_train \
    --val_dir /media/zhangxin/data1/data_public/tianchi/复赛/imgs2_val \
    --train_file /media/zhangxin/data1/data_public/tianchi/复赛/train2.txt \
    --val_file /media/zhangxin/data1/data_public/tianchi/复赛/val2.txt \
    --val_num 100

# 3 训练检测模型

## 3.1 icdar 2015
python3 tools/train.py -c configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.icdar2015.yml
visualdl --logdir output/ch_db_mv3.icdar2015/vdl -p 8080 -t 10.168.47.17

export CUDA_VISIBLE_DEVICES='0'
nohup python3 tools/train.py \
    -c configs/det/det_r50_vd_db_zx.invoice.yml \
    -o Global.pretrain_weights=./inference/ch_ppocr_server_v2.0_det_train/ \
    >nohup.20210712_0650.train.det_r50_vd_db_zx.invoice.out &




