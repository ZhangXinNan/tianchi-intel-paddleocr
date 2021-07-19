import random

import pandas as pd
import json
import os
import urllib
# from urllib import request
import urllib.request
from tqdm import tqdm
import numpy as np
import cv2


def down_image(url, out_dir):
    print(url)
    save_path = os.path.join(out_dir, os.path.basename(url))
    # if os.path.exists('./train_data/tianchi/image/' + url.split('/')[-1]):
    #     return
    if os.path.isfile(save_path):
        return
    # urllib.request.urlretrieve(url, './train_data/tianchi/image/' + url.split('/')[-1])
    urllib.request.urlretrieve(url, save_path)


def convert_tianchi_paddleocr(annot, img):
    if rotate_by_option:
        h, w = img.shape[:2]
        if annot[1] == '底部朝左':
            img = cv2.flip(cv2.transpose(img), 0)
        elif annot[1] == '底部朝上':
            img = cv2.flip(img, -1)
        elif annot[1] == '底部朝右':
            img = cv2.flip(cv2.transpose(img), 1)

    annot_ppocr = []
    for text_line in annot[0]:
        transcription = json.loads(text_line['text'])['text']
        pts = np.array([float(x) for x in text_line['coord']]).reshape((4, 2)).tolist()
        if rotate_by_option:
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            x3, y3 = pts[2]
            x4, y4 = pts[3]
            if annot[1] == '底部朝左':
                pts = [[h - 1 - y4, x4], [h - 1 - y1, x1],
                       [h - 1 - y2, x2], [h - 1 - y3, x3]]
            elif annot[1] == '底部朝上':
                pts = [[w - 1 - x3, h - 1 - y3], [w - 1 - x4, h - 1 - y4],
                       [w - 1 - x1, h - 1 - y1], [w - 1 - x2, h - 1 - y2]]
            elif annot[1] == '底部朝右':
                pts = [[y2, w - 1 - x2], [y3, w - 1 - x3],
                       [y4, w - 1 - x4], [y1, w - 1 - x1]]
        annot_ppocr.append({"transcription": transcription, "points": pts, "difficult": False})
    return json.dumps(annot_ppocr, ensure_ascii=False), img


def process_data(csv_path, img_dir, src_img_dir, dst_img_dir):
    '''获取训练数据'''
    in_img_dir = os.path.join(src_img_dir, img_dir)
    out_img_dir = os.path.join(dst_img_dir, img_dir)
    if not os.path.isdir(out_img_dir):
        os.makedirs(out_img_dir)

    result = []
    for row in pd.read_csv(csv_path).iterrows():
        img_url = json.loads(row[1]['原始数据'])['tfspath']
        out_img_path = os.path.join(out_img_dir, os.path.basename(img_url))
        src_img_path = os.path.join(in_img_dir, os.path.basename(img_url))
        if '融合答案' in row[1]:
            annot = json.loads(row[1]['融合答案'])
            img = cv2.imread(src_img_path, cv2.IMREAD_COLOR)
            annot, img = convert_tianchi_paddleocr(annot, img)
            cv2.imwrite(out_img_path, img)
            result.append([out_img_path, annot])
    return result


def split_train_val(result, out_train_file, out_val_file, val_num):
    random.shuffle(result)
    out_dir = os.path.dirname(out_train_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fo_val = open(out_val_file, 'w')
    fo_train = open(out_train_file, 'w')
    for i, one in enumerate(result):
        if i < val_num:
            fo_val.write("{}\t{}\n".format(one[0], one[1]))
        else:
            fo_train.write("{}\t{}\n".format(one[0], one[1]))
    fo_train.close()
    fo_val.close()
    pass


def main(args):
    global rotate_by_option
    rotate_by_option = args.rotate_by_option.lower() == 'true'
    print(rotate_by_option)

    csv_files = [
        ('Xeon1OCR_round1_train1_20210526.csv', 'train1', 49),
        ('Xeon1OCR_round1_train2_20210526.csv', 'train2', 441),
        ('Xeon1OCR_round1_train_20210524.csv', 'train', 135)
    ]
    for name, img_dir, val_num in csv_files:
        result = process_data('input/' + name, img_dir, args.src_img_dir, args.dst_img_dir)
        split_train_val(result,
                        '{}/det_data/{}.train.txt'.format(args.dst_img_dir, name),
                        '{}/det_data/{}.val.txt'.format(args.dst_img_dir, name),
                        val_num)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rotate_by_option', default='True')
    parser.add_argument('--src_img_dir', default='./data')
    parser.add_argument('--dst_img_dir', default='./data_calib')
    return parser.parse_args()


if __name__ == '__main__':
    import argparse
    rotate_by_option = True
    main(get_args())
