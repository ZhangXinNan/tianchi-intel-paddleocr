
8363 OCR复赛数据集01.csv        npx
2594 OCR复赛数据集02.csv        invoice

    7962 OCR复赛数据集01.csv.train.txt
     400 OCR复赛数据集01.csv.val.txt
    2493 OCR复赛数据集02.csv.train.txt
     100 OCR复赛数据集02.csv.val.txt
   10955 total


export CUDA_VISIBLE_DEVICES='1'
nohup python3 tools/train.py \
    -c configs/det/det_r50_vd_db_zx.invoice2.yml \
    -o Global.pretrain_weights=./inference/ch_ppocr_server_v2.0_det_train/ \
    >nohup.20210903.train.det_r50_vd_db_zx.invoice2.out &
visualdl --logdir ./output/det_r50_vd_zx.invoice2/vdl -p 8081 -t 192.168.200.7



export CUDA_VISIBLE_DEVICES='0'
nohup python3 tools/train.py \
    -c configs/det/det_r50_vd_db_zx.npx2.yml \
    -o Global.pretrain_weights=./inference/ch_ppocr_server_v2.0_det_train/ Optimizer.base_lr=0.0001 \
    >nohup.20210903.train.det_r50_vd_db_zx.npx2.out &
visualdl --logdir ./output/det_r50_vd_zx.npx2/vdl -p 8082 -t 192.168.200.7


export CUDA_VISIBLE_DEVICES='1'
nohup python3 tools/train.py \
    -c configs/det/det_r50_vd_db_zx.invoice+npx2.yml \
    -o Optimizer.base_lr=0.0001 \
    >nohup.20210903.train.det_r50_vd_db_zx.invoice+npx2.out &
visualdl --logdir ./output/det_r50_vd_zx.invoice+npx2/vdl -p 8083 -t 192.168.200.7