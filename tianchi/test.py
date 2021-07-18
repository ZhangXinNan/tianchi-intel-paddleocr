
import os
import json
import numpy as np
import cv2


def points_str2int(pts):
    pts_f = []
    for pt in pts:
        pts_f.append([int(float(pt[0])), int(float(pt[1]))])
    return pts_f


# filename = 'TB2OIJ1dOMnBKNjSZFCXXX0KFXa_%21%216000000005029-0-cloudwork.jpg'
# filename = 'TB2RcwudIIrBKNjSZK9XXagoVXa_%21%216000000006696-0-cloudwork.jpg'
filename = 'TB2AXcxe8yWBuNkSmFPXXXguVXa_%21%216000000003409-0-cloudwork.jpg'
print(filename)

img_dir = './train/'

# content = open(filename, 'r').read()
# print(content)

# annot = json.loads(content)
# print(annot)

train_file = '/home/zhangxin/github/tianchi-intel-PaddleOCR/data/det_data/Xeon1OCR_round1_train_20210524.csv.train.txt'
with open(train_file, 'r') as fi:
    for line in fi:
        arr = line.strip().split('\t')
        if len(arr) < 2:
            continue
        if filename[:-4] in arr[0]:
            print(arr)
            annot = json.loads(arr[1])
            print(annot)
            print(len(annot))
            

            img = cv2.imread(os.path.join(img_dir, os.path.basename(arr[0])), cv2.IMREAD_COLOR)
            img_bak = img.copy()
            for i, one in enumerate(annot):
                pts = points_str2int(one['points'])
                print(i, len(one['points']), one, pts)
                cv2.fillConvexPoly(img, np.array(pts), (0, 255, 0))
            print(img.shape)
            img = cv2.addWeighted(img, 0.5, img_bak, 0.5, 0)
            cv2.imwrite(os.path.basename(arr[0]), img)
            break

