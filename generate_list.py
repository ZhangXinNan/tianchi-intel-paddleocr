import os, sys, json
import cv2
import numpy as np
import pandas as pd

IMAGE_PATH = './input/'
train = pd.concat([
    pd.read_csv('input/Xeon1OCR_round1_train1_20210526.csv'),
    pd.read_csv('input/Xeon1OCR_round1_train_20210524.csv')]
)

train = train.sample(frac=1.0)

idx = 0
for row in train.iloc[:-200].iterrows():
    path = json.loads(row[1]['原始数据'])['tfspath']
    img_path = IMAGE_PATH + path.split('/')[-1]
    labels = json.loads(row[1]['融合答案'])[0]
    
    ann_text = []
    for label in labels[:]:
        text = json.loads(label['text'])['text']
        coord = [int(float(x)) for x in label['coord']]
        
        ann_text.append({
            'transcription': text,
            'points': [coord[:2], coord[2:4],coord[4:6], coord[-2:]]
        })
        # break
    with open('./train_data/tianchi/train_list.txt', 'a+') as up:
        up.write(f"image/{path.split('/')[-1]}\t{json.dumps(ann_text)}\n")

idx = 0
for row in train.iloc[-200:].iterrows():
    path = json.loads(row[1]['原始数据'])['tfspath']
    img_path = IMAGE_PATH + path.split('/')[-1]
    labels = json.loads(row[1]['融合答案'])[0]
    
    ann_text = []
    for label in labels[:]:
        text = json.loads(label['text'])['text']
        coord = [int(float(x)) for x in label['coord']]
        
        ann_text.append({
            'transcription': text,
            'points': [coord[:2], coord[2:4],coord[4:6], coord[-2:]]
        })
        # break
    with open('./train_data/tianchi/test_list.txt', 'a+') as up:
        up.write(f"image/{path.split('/')[-1]}\t{json.dumps(ann_text)}\n")