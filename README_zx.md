
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
