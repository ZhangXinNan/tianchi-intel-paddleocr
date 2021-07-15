


# test1 book

python3 tools/infer/predict_system_tianchi.py \
    --det_model_dir="./inference/ch_ppocr_server_v2.0_det_infer"  \
    --rec_model_dir="./inference/ch_ppocr_server_v2.0_rec_infer/" \
    --cls_model_dir='./inference/ch_ppocr_mobile_v2.0_cls_infer/' \
    --use_angle_cls=True \
    --use_space_char=True \
    --test_csvfile input/Xeon1OCR_round1_test1_20210528.csv \
    --test_img_dir data/test1/ \
    --test_result Xeon1OCR_round1_test1_20210528.json

# test2 invoice

python3 tools/export_model.py \
    -c configs/det/det_r50_vd_db_zx.invoice.yml \
    -o Global.pretrained_model=output/det_r50_vd_zx.invoice/best_accuracy  Global.save_inference_dir=output/det_r50_vd_zx.invoice/

python3 tools/infer/predict_system_tianchi.py \
    --det_model_dir="./output/det_r50_vd_zx.invoice"  \
    --rec_model_dir="./output/rec_ch_20210713" \
    --cls_model_dir='./inference/ch_ppocr_mobile_v2.0_cls_infer/' \
    --use_angle_cls=True \
    --use_space_char=True \
    --test_csvfile input/Xeon1OCR_round1_test2_20210528.csv \
    --test_img_dir data/test2/ \
    --test_result Xeon1OCR_round1_test2_20210528.json

# test npx

python3 tools/export_model.py \
    -c configs/det/det_r50_vd_db_zx.npx.yml \
    -o Global.pretrained_model=output/det_r50_vd_zx.npx/best_accuracy  Global.save_inference_dir=output/det_r50_vd_zx.npx/

python3 tools/infer/predict_system_tianchi.py \
    --det_model_dir="./output/det_r50_vd_zx.npx"  \
    --rec_model_dir="./output/rec_ch_20210713" \
    --cls_model_dir='./inference/ch_ppocr_mobile_v2.0_cls_infer/' \
    --use_angle_cls=True \
    --use_space_char=True \
    --test_csvfile input/Xeon1OCR_round1_test3_20210528.csv \
    --test_img_dir data/test3/ \
    --test_result Xeon1OCR_round1_test3_20210528.json
