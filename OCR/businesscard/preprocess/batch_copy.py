
import os
import cv2
import shutil

src_path = './dst_clean_version_test_5.7k/'
dst_path = './json/'

imagelist = os.listdir(src_path)

for imgname in imagelist:
    if(imgname.endswith(".json")):
        print(imgname)
        #img_name.split('_')[1]
        shutil.copy(src_path +imgname, dst_path+imgname)
