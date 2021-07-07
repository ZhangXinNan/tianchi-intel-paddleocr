# “英特尔创新大师杯”深度学习挑战赛 赛道1：通用场景OCR文本识别任务

https://tianchi.aliyun.com/competition/entrance/531902/introduction

## 环境配置

### paddlepaddle

安装`paddlepaddle-gpu`，在安装好CUDA的情况下，可以直接通过pip安装。

官方安装文档：https://www.paddlepaddle.org.cn/install/quick

### 其他环境

- python3
- 需要有GPU

## 总步骤：一键训练预测

为了方便大家运行baseline，这里写好了训练和预测代码，在GPU的情况下需要训练半个小时，然后10分钟预测。

```
git clone https://gitee.com/coggle/tianchi-intel-PaddleOCR
cd tianchi-intel-PaddleOCR
sh run.sh
```

当然也可以分步骤执行，参考下面的教程。注意下面教程都在代码根目录执行。

## 步骤1：下载比赛图片

```
python3 down_image.py
```

保存目录为`train_data/tianchi/image`，按照文件名进行保存，训练集和测试集存储在一起。

## 步骤2：下载预测模型

由于OCR包括多个步骤，此时我们只对其中检测的部署进行fientune，所以其他部署的权重也需要下载。

```
mkdir inference && cd inference/

# 下载模型
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar

# 解压模型
tar -xf ch_ppocr_server_v2.0_rec_infer.tar 
tar -xf ch_ppocr_server_v2.0_det_infer.tar
tar -xf ch_ppocr_mobile_v2.0_cls_infer.tar
```

下载完成后可以验证是否可以成功预测：

```
python3 tools/infer/predict_system.py --image_dir="./1.jpg" --det_model_dir="./inference/ch_ppocr_server_v2.0_det_infer/"  --rec_model_dir="./inference/ch_ppocr_server_v2.0_rec_infer/" --cls_model_dir='./inference/ch_ppocr_mobile_v2.0_cls_infer/' --use_angle_cls=True --use_space_char=True
```

输出结果为：

```
dt_boxes num : 2, elapse : 0.9568207263946533
cls num  : 2, elapse : 0.006417512893676758
rec_res num  : 2, elapse : 0.05788707733154297
Predict time of ./1.jpg: 1.036s
土地整治与土壤修复研究中心, 0.973
华南农业大学-东图, 0.992
```

如果直接使用预训练模型，其实也可以得到不错的分数。但是比赛数据集与通用数据集存在差异，finetune后精度会更好。

## 步骤3：训练预检测模型

首先下载检测模块的预训练模型：

```
cd inference
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar
tar -xf ch_ppocr_server_v2.0_det_train.tar
```

然后进行finetune，这里训练4个epoch，30分钟左右完成训练。

```
python3 tools/train.py -c configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml -o Global.pretrain_weights=./inference/ch_ppocr_server_v2.0_det_train/
```

## 步骤4：对测试集进行预测

训练完成后，接下来需要将模型权重导出，用于预测。并对测试集的图片进行预测，写入json。

```
# 将模型导出
python3 tools/export_model.py -c configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml -o Global.pretrained_model=output/ch_db_res18/best_accuracy  Global.save_inference_dir=output/ch_db_res18/

# 对测试集进行预测
python3 tools/infer/predict_system_tianchi.py --image_dir="./doc/imgs/11.jpg" --det_model_dir="output/ch_db_res18/"  --rec_model_dir="./inference/ch_ppocr_server_v2.0_rec_infer/" --cls_model_dir='./inference/ch_ppocr_mobile_v2.0_cls_infer/' --use_angle_cls=True --use_space_char=True

# 将结果文件压缩
zip -r submit.zip Xeon1OCR_round1_test*
```

## 参考资料

https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_ch/customize.md

https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_ch/recognition.md

https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_ch/inference.md


公众号二维码
![coggle](https://coggle.club/assets/img/coggle_qrcode.jpg)