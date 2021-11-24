
import os
import cv2
import shutil

src_path = './train_images/'

imagelist = os.listdir(src_path)

f=open("train_list.txt", "a+")

for imgname in imagelist:
    if(imgname.endswith(".jpg")):
        print(imgname)
        result_str = imgname+'\n'
        f.write(result_str)
