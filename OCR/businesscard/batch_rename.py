import os
from PIL import Image


path_name='./qr_samples/'
#path_name :表示你需要批量改的文件夹
i=0
dst_name = ''

def is_valid(file):
    valid = True
    try:
        Image.open(file).load()
    except OSError:
        valid = False
    return valid

for item in os.listdir(path_name):#进入到文件夹内，对每个文件进行循环遍历
    img_valid = is_valid(path_name+item)
    if img_valid:
        os.rename(os.path.join(path_name,item),os.path.join(path_name,(str(i)+'.jpg')))#os.path.join(path_name,item)表示找到每个文件的绝对路径并进行拼接操作
        i+=1
    else:
        print(path_name+item)
        os.remove(path_name+item)
