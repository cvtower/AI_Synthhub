import cv2
import numpy as np
import shutil
import os
import random

src_dir = 'test2017/'
dst_dir = 'total_bg/'
image_list = os.listdir(src_dir)
target_cnt = 6000

image_items = random.sample(image_list, target_cnt)

for image_item in image_items:
    if image_item.split('.')[-1] != 'jpg':
        continue
    print(image_item)
    img_path = src_dir + image_item
    dst_path = dst_dir + image_item
    shutil.copy(img_path, dst_path)