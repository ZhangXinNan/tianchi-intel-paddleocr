import random

import pandas as pd
import json
import os
import urllib
# from urllib import request
import urllib.request
from tqdm import tqdm
'''
urls = []

# 训练集
train = pd.concat([
    pd.read_csv('input/Xeon1OCR_round1_train1_20210526.csv'),
    pd.read_csv('input/Xeon1OCR_round1_train_20210524.csv')]
)

try:
    os.mkdir('./train_data/tianchi/image/')
except:
    pass

for row in train.iterrows():
    path = json.loads(row[1]['原始数据'])['tfspath']
    urls.append(path)
    
# 测试集
train = pd.concat([
    pd.read_csv('input/Xeon1OCR_round1_test1_20210528.csv'),
    pd.read_csv('input/Xeon1OCR_round1_test2_20210528.csv'),
    pd.read_csv('input/Xeon1OCR_round1_test3_20210528.csv')]
)

for row in train.iterrows():
    path = json.loads(row[1]['原始数据'])['tfspath']
    urls.append(path)

print('Total images: ', len(urls))
'''


def down_image(url, out_dir):
    print(url)
    save_path = os.path.join(out_dir, os.path.basename(url))
    # if os.path.exists('./train_data/tianchi/image/' + url.split('/')[-1]):
    #     return
    if os.path.isfile(save_path):
        return
    # urllib.request.urlretrieve(url, './train_data/tianchi/image/' + url.split('/')[-1])
    urllib.request.urlretrieve(url, save_path)


# from joblib import Parallel, delayed
# Parallel(n_jobs=-1)(delayed(down_image)(url) for url in tqdm(urls))


# for row in train.iterrows():
#     path = json.loads(row[1]['原始数据'])['tfspath']
#     print(path)
#     if os.path.exists('./train_data/tianchi/image/' + path.split('/')[-1]):
#         continue
#     urllib.request.urlretrieve(path, './train_data/tianchi/image/' + path.split('/')[-1])
def process_data(csv_path, out_img_dir):
    '''获取训练数据'''
    if not os.path.isdir(out_img_dir):
        os.makedirs(out_img_dir)

    result = []
    for row in pd.read_csv(csv_path).iterrows():
        img_url = json.loads(row[1]['原始数据'])['tfspath']
        img_path = os.path.join(out_img_dir, os.path.basename(img_url))
        if not os.path.isfile(img_path):
            urllib.request.urlretrieve(img_url, img_path)
        if '融合答案' in row[1]:
            annot = json.loads(row[1]['融合答案'])
            result.append([img_path, annot])
    return result


def split_train_val(result, out_train_file, out_val_file, val_num):
    # random.shuffle(result)
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


def main():
    csv_files = [('Xeon1OCR_round1_train1_20210526.csv', 'data/train1', 49),
                 ('Xeon1OCR_round1_train2_20210526.csv', 'data/train2', 441),
                 ('Xeon1OCR_round1_train_20210524.csv', 'data/train', 135)]
    for name, img_dir, val_num in csv_files:
        result = process_data('input/' + name, img_dir)
        split_train_val(result,
                        'data/det_data/{}.train.txt'.format(name),
                        'data/det_data/{}.val.txt'.format(name), val_num)

    csv_files = [('Xeon1OCR_round1_test1_20210528.csv', 'data/test1'),
                 ('Xeon1OCR_round1_test2_20210528.csv', 'data/test2'),
                 ('Xeon1OCR_round1_test3_20210528.csv', 'data/test3')]
    for name, img_dir in csv_files:
        process_data('input/' + name, img_dir)


if __name__ == '__main__':
    main()
