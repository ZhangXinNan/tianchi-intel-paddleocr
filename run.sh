# 下载图片
python3 down_image.py

mkdir inference && cd inference/

# 下载预测模型
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar

# 解压模型
tar -xf ch_ppocr_server_v2.0_rec_infer.tar 
tar -xf ch_ppocr_server_v2.0_det_infer.tar
tar -xf ch_ppocr_mobile_v2.0_cls_infer.tar

cd ..

# 测试已有模型
python3 tools/infer/predict_system.py --image_dir="./1.jpg" --det_model_dir="./inference/ch_ppocr_server_v2.0_det_infer/"  --rec_model_dir="./inference/ch_ppocr_server_v2.0_rec_infer/" --cls_model_dir='./inference/ch_ppocr_mobile_v2.0_cls_infer/' --use_angle_cls=True --use_space_char=True

# 下载预训练模型
cd inference
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar
tar -xf ch_ppocr_server_v2.0_det_train.tar

cd ..
# 进行训练
python3 tools/train.py -c configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml -o Global.pretrain_weights=./inference/ch_ppocr_server_v2.0_det_train/

# 导出模型
python3 tools/export_model.py -c configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml -o Global.pretrained_model=output/ch_db_res18/best_accuracy  Global.save_inference_dir=output/ch_db_res18/

# 测试集预测
python3 tools/infer/predict_system_tianchi.py --det_model_dir="output/ch_db_res18/"  --rec_model_dir="./inference/ch_ppocr_server_v2.0_rec_infer/" --cls_model_dir='./inference/ch_ppocr_mobile_v2.0_cls_infer/' --use_angle_cls=True --use_space_char=True

# 压缩结果文件
zip -r submit.zip Xeon1OCR_round1_test*