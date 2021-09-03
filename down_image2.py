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


def convert_tianchi_paddleocr(annot, img_path):
    if rotate_by_option:
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        if annot[1] == '底部朝左':
            img = cv2.flip(cv2.transpose(img), 0)
            cv2.imwrite(img_path, img)
        elif annot[1] == '底部朝上':
            img = cv2.flip(img, -1)
            cv2.imwrite(img_path, img)
        elif annot[1] == '底部朝右':
            img = cv2.flip(cv2.transpose(img), 1)
            cv2.imwrite(img_path, img)

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
    return json.dumps(annot_ppocr, ensure_ascii=False)


def process_data(csv_path, out_img_dir):
    '''获取训练数据'''
    if not os.path.isdir(out_img_dir):
        os.makedirs(out_img_dir)

    result = []
    for row in pd.read_csv(csv_path).iterrows():
        img_url = json.loads(row[1]['原始数据'])['tfspath']
        img_path = os.path.join(out_img_dir, os.path.basename(img_url))
        # if not os.path.isfile(img_path):
        urllib.request.urlretrieve(img_url, img_path)
        if '融合答案' in row[1]:
            annot = json.loads(row[1]['融合答案'])
            annot = convert_tianchi_paddleocr(annot, img_path)
            result.append([img_path, annot])
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
    '''
    csv_files = [('Xeon1OCR_round1_train1_20210526.csv', 'data/train1', 49),
                 ('Xeon1OCR_round1_train2_20210526.csv', 'data/train2', 441),
                 ('Xeon1OCR_round1_train_20210524.csv', 'data/train', 135)]
    for name, img_dir, val_num in csv_files:
        result = process_data('input/' + name, img_dir)
        split_train_val(result,
                        '{}/{}.train.txt'.format(args.train_val_dir, name),
                        '{}/{}.val.txt'.format(args.train_val_dir, name), val_num)

    csv_files = [('Xeon1OCR_round1_test1_20210528.csv', 'data/test1'),
                 ('Xeon1OCR_round1_test2_20210528.csv', 'data/test2'),
                 ('Xeon1OCR_round1_test3_20210528.csv', 'data/test3')]
    for name, img_dir in csv_files:
        process_data('input/' + name, img_dir)
    '''
    csv_files = [('OCR复赛数据集01.csv', 'data/train1', 400), ('OCR复赛数据集02.csv', 'data/train2', 100)]
    for name, img_dir, val_num in csv_files:
        result = process_data('input/' + name, img_dir)
        split_train_val(result,
                        '{}/{}.train.txt'.format(args.train_val_dir, name),
                        '{}/{}.val.txt'.format(args.train_val_dir, name), val_num)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_val_dir', default='data/det_data2')
    parser.add_argument('--rotate_by_option', default='False')
    return parser.parse_args()


if __name__ == '__main__':
    import argparse
    rotate_by_option = False
    main(get_args())
