
import os
import cv2
import shutil

src_path = './test_gts/'

imagelist = os.listdir(src_path)

for imgname in imagelist:
    if(imgname.endswith(".txt")):
        print(imgname)
        #file_name = imgname.split('.')[0]
        #new_file_name = file_name.split('_')[2]
        #new_name = file_name + '.jpg.txt'
        new_name = 'gt_'+ imgname
        shutil.move(src_path +imgname, src_path + new_name)
