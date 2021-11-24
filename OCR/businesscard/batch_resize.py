# coding=utf-8
import os
import cv2
import shutil
 
src_dir = 'total_bg/'
dst_dir = 'resized_debug/'
imagelist = os.listdir(src_dir)

dst_size = 224

for imgname in imagelist:
    #print(imgname)
    if (imgname.endswith(".jpg")) or (imgname.endswith(".jpeg") ) or (imgname.endswith(".png") ):
        try:
            image = cv2.imread(src_dir +imgname)
            sp = image.shape
            img_w = sp[1]
            img_h = sp[0]
           
            resized_image = cv2.resize(image, (dst_size, dst_size))
        #print(img_w, img_h, new_w, new_h)
        #print()
            cv2.imwrite(dst_dir + 'hl_' + imgname, resized_image)
        except  Exception as e:
            print(src_dir +imgname)
            os.remove(src_dir + imgname)
    else:
        continue
