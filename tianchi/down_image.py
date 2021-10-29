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


def main(args):
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    for row in pd.read_csv(args.input).iterrows():
        img_url = json.loads(row[1]['原始数据'])['tfspath']
        img_path = os.path.join(args.out_dir, os.path.basename(img_url))
        if os.path.isfile(img_path):
            print(img_path, '【exists】')
            continue
        urllib.request.urlretrieve(img_url, img_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='../input/复赛/OCR复赛数据集01.csv')
    parser.add_argument('--out_dir', default='../data/imgs1')
    return parser.parse_args()


if __name__ == '__main__':
    import argparse
    main(get_args())
