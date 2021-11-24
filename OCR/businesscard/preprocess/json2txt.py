# coding:utf-8

import os
import json
import numpy as np

def order_points_new(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    if leftMost[0,1]!=leftMost[1,1]:
        leftMost=leftMost[np.argsort(leftMost[:,1]),:]
    else:
        leftMost=leftMost[np.argsort(leftMost[:,0])[::-1],:]
    (tl, bl) = leftMost
    if rightMost[0,1]!=rightMost[1,1]:
        rightMost=rightMost[np.argsort(rightMost[:,1]),:]
    else:
        rightMost=rightMost[np.argsort(rightMost[:,0])[::-1],:]
    (tr,br)=rightMost
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def json2txt(path_json,path_txt):
    with open(path_json,'r', encoding='gb18030') as path_json:
        jsonx=json.load(path_json)
        with open(path_txt,'w+') as ftxt:
            for shape in jsonx['shapes']:
                #print(shape['points'])
                sorted_points = order_points_new(np.array(shape['points']))
                xy=np.array(sorted_points)
                
                label=str(shape['label'])
                strxy = ''
                for m,n in xy:
                    strxy+=str(m)+','+str(n)+','
                strxy+=label
                ftxt.writelines(strxy+"\n")

dir_json = 'json/'
dir_txt = 'test_gts/'
if not os.path.exists(dir_txt):
    os.makedirs(dir_txt)
list_json = os.listdir(dir_json)
for cnt,json_name in enumerate(list_json):
    print('cnt=%d,name=%s'%(cnt,json_name))
    path_json = dir_json + json_name
    path_txt = dir_txt + json_name.replace('.json','.txt')
    # print(path_json, path_txt)
    json2txt(path_json, path_txt)
    #exit()
