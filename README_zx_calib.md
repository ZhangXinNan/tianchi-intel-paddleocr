
# 训练
export CUDA_VISIBLE_DEVICES='0'
nohup python3 tools/train.py \
    -c configs/det/det_r50_vd_db_zx.calib.npx.yml \
    >nohup.train.det_r50_vd_db_zx.calib.npx.out &
visualdl --logdir output/det_r50_vd_db_zx.calib.npx/vdl -p 8080 -t 10.168.12.11





export CUDA_VISIBLE_DEVICES='1'
nohup python3 tools/train.py \
    -c configs/det/det_r50_vd_db_zx.calib.invoice.yml \
    >nohup.train.det_r50_vd_db_zx.calib.invoice.out &
visualdl --logdir output/det_r50_vd_db_zx.calib.invoice/vdl -p 8081 -t 10.168.12.11



# 测试

## invoice
CONFIG_FILE=configs/det/det_r50_vd_db_zx.calib.invoice.yml
MODLE_DIR=output/det_r50_vd_db_zx.calib.invoice
python3 tools/export_model.py \
    -c configs/det/det_r50_vd_db_zx.calib.invoice.yml \
    -o Global.pretrained_model=output/det_r50_vd_db_zx.calib.invoice/best_accuracy  Global.save_inference_dir=output/det_r50_vd_db_zx.calib.invoice


python3 tools/infer/predict_system_tianchi.py \
    --det_model_dir="./output/det_r50_vd_db_zx.calib.invoice"  \
    --rec_model_dir="./output/rec_ch_20210713" \
    --cls_model_dir='./inference/ch_ppocr_mobile_v2.0_cls_infer/' \
    --use_angle_cls=True \
    --use_space_char=True \
    --test_csvfile input/Xeon1OCR_round1_test2_20210528.csv \
    --test_img_dir data/test2/ \
    --test_result Xeon1OCR_round1_test2_20210528.json

python3 tools/export_model.py \
    -c configs/det/det_r50_vd_db_zx.calib.npx.yml \
    -o Global.pretrained_model=output/det_r50_vd_db_zx.calib.npx/best_accuracy  Global.save_inference_dir=output/det_r50_vd_db_zx.calib.npx

python3 tools/infer/predict_system_tianchi.py \
    --det_model_dir="./output/det_r50_vd_db_zx.calib.npx"  \
    --rec_model_dir="./output/rec_ch_20210713" \
    --cls_model_dir='./inference/ch_ppocr_mobile_v2.0_cls_infer/' \
    --use_angle_cls=True \
    --use_space_char=True \
    --test_csvfile input/Xeon1OCR_round1_test3_20210528.csv \
    --test_img_dir data/test3/ \
    --test_result Xeon1OCR_round1_test3_20210528.json