
# 数据
book    train1  868,49,918      test1   49
invoice train   2458,135,2594   test2   135
npx     train2  7921,441,8363   test3   441



```bash

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
epoch: [390/1200], iter: 60000, lr: 0.001000
best metric, hmean: 0.6432003137562505, precision: 0.7324698526127735, recall: 0.5733263415486803, fps: 7.811742308914901, best_epoch: 234


export CUDA_VISIBLE_DEVICES='1'
nohup python3 tools/train.py \
    -c configs/det/ch_det_res18_db.zx.invoice.yml \
    >nohup.train.ch_det_res18_db.zx.invoice.out &
visualdl --logdir output/ch_db_res18.zx.invoice/vdl -p 8081 -t 192.168.144.125

# 第二次训练使用invoice + npx
export CUDA_VISIBLE_DEVICES='1'
nohup python3 tools/train.py \
    -c configs/det/det_r50_vd_db_zx.invoice+npx.yml \
    -o Global.pretrain_weights=./inference/ch_ppocr_server_v2.0_det_train/ Optimizer.base_lr=0.0001 \
    >nohup.train.det_r50_vd_db_zx.invoice+npx.out &
step    50000   hmean   0.57062e


export CUDA_VISIBLE_DEVICES='2'
nohup python3 tools/train.py \
    -c configs/det/ch_det_res18_db.zx.invoice+npx.yml \
    >nohup.train.ch_det_res18_db.zx.invoice+npx.out &
visualdl --logdir output/ch_db_res18.zx.invoice+npx/vdl -p 8082 -t 192.168.144.125

# 第三次训练npx
export CUDA_VISIBLE_DEVICES='0'
nohup python3 tools/train.py \
    -c configs/det/det_r50_vd_db_zx.npx.yml \
    -o Global.pretrain_weights=./inference/ch_ppocr_server_v2.0_det_train/ Optimizer.base_lr=0.0001 \
    >nohup.20210712_1850.train2.det_r50_vd_db_zx.npx.out &
step    54000   hmean   0.58004
epoch: [351/1200], iter: 174000, lr: 0.000100
best metric, hmean: 0.6432003137562505, precision: 0.7324698526127735, recall: 0.5733263415486803



export CUDA_VISIBLE_DEVICES='3'
nohup python3 tools/train.py \
    -c configs/det/ch_det_res18_db.zx.npx.yml \
    >nohup.train.ch_det_res18_db.zx.npx.out &
visualdl --logdir output/ch_db_res18.zx.npx/vdl -p 8083 -t 192.168.144.125
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
