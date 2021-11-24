# -*- coding:utf-8 -*-

import os
import csv
import numpy as np
import cv2
import shutil
import random
from os.path import join as pjoin
import json
import base64
from PIL import Image, ImageFont, ImageDraw

SRC = "./bcard/"
SRC_BG = "./test2017/"
DST = "./img_dst/"
skip_list = []
skipbg_list = []
global sync_cnt
target_cnt = 100000
repeat_time = 1
state = 0

global label
enable_debug = False
def rad(x):
    return x*np.pi/180
def dict_json(version,imageData,shapes,imagePath,height,width,fillColor=None,lineColor=None):
    '''

    :param imageData: str
    :param shapes: list
    :param imagePath: str
    :param fillColor: list
    :param lineColor: list
    :return: dict
    '''
    return {"version":version,"imageData":imageData,"shapes":shapes,"fillColor":fillColor,
            'imagePath':imagePath,'lineColor':lineColor,'imageHeight': height,'imageWidth': width, }

def dict_shapes(points,label,fill_color=None,line_color=None):
    return {'points':points,'fill_color':fill_color,'label':label,'line_color':line_color}

def synth_img(fg_img_ori, bg_img_ori, src_quad_list):

    dst_quad_list= []
    fg_height,fg_width= fg_img_ori.shape[0:2]
    #print('fg_height,fg_width:', fg_height,fg_width)
    '''if fg_height>fg_width:
        bg_height = 1280
        bg_width = 720
    else:
        bg_height = 720
        bg_width = 1280'''
    bg_height = 736
    bg_width = 736
    #print('bg_width,bg_height:', bg_width, bg_height)
    try:
        bg_img = cv2.resize(bg_img_ori, dsize=(int(bg_width), int(bg_height)))
    except  Exception as e:
        print('Exception while executing: bg_img = cv2.resize')
        exit()
        return False, 0, 0
    bg_height,bg_width=bg_img.shape[0:2]

    fg_ori_height,fg_ori_width=fg_img_ori.shape[0:2]
    fg_scale = random.uniform(0.4, 0.9)
    fg_width = int(fg_scale*bg_width)
    try:
        fg_img = cv2.resize(fg_img_ori, dsize=(int(fg_width), int(fg_ori_height*fg_width/fg_ori_width)))
    except  Exception as e:
        print('Exception while executing: fg_img = cv2.resize')
        exit()
        return False, 0, 0
    fg_height,fg_width=fg_img.shape[0:2]

    dx_left = 0
    dx_right = 0
    dy_up = 0
    dy_down = 0
    if bg_width<=fg_width and bg_height<=fg_height:
        print('ERROR: bg image too small!...should not happen')
        return False, 0, 0

    anglex = random.uniform(-25, 25)
    angley = random.uniform(-25, 25)
    anglez = random.uniform(-180, 180)
    fov = 50
    z_value = random.randint(-10, 0)

    z=np.sqrt(bg_width**2 + bg_height**2)/2/np.tan(rad(fov/2))

    rx = np.array([[1,                  0,                          0,                          0],
                   [0,                  np.cos(rad(anglex)),        -np.sin(rad(anglex)),       0],
                   [0,                 -np.sin(rad(anglex)),        np.cos(rad(anglex)),        0,],
                   [0,                  0,                          0,                          1]], np.float32)

    ry = np.array([[np.cos(rad(angley)), 0,                         np.sin(rad(angley)),       0],
                   [0,                   1,                         0,                          0],
                   [-np.sin(rad(angley)),0,                         np.cos(rad(angley)),        0,],
                   [0,                   0,                         0,                          1]], np.float32)

    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)),      0,                          0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)),      0,                          0],
                   [0,                  0,                          1,                          0],
                   [0,                  0,                          0,                          1]], np.float32)

    r = rx.dot(ry).dot(rz)

    #4 pair vertex gen
    pcenter = np.array([int(bg_width/2), int(bg_height/2), 0, 0], np.float32)

    p1 = np.array([(bg_width-fg_width)/2,(bg_height-fg_height)/2,  z_value,0], np.float32) - pcenter
    p2 = np.array([(bg_width+fg_width)/2,(bg_height-fg_height)/2,  z_value,0], np.float32) - pcenter
    p3 = np.array([(bg_width+fg_width)/2,(bg_height+fg_height)/2,  z_value,0], np.float32) - pcenter
    p4 = np.array([(bg_width-fg_width)/2,(bg_height+fg_height)/2,  z_value,0], np.float32) - pcenter

    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)

    list_dst = [dst1, dst2, dst3, dst4]

    org = np.array([[0,0],
                    [fg_width,0],
                    [fg_width,fg_height],
                    [0,fg_height]], np.float32)
    
    dst = np.zeros((4,2), np.float32)

    #projection transform
    for i in range(4):
        dst[i,0] = list_dst[i][0]*z/(z-list_dst[i][2]) + pcenter[0]
        dst[i,1] = list_dst[i][1]*z/(z-list_dst[i][2]) + pcenter[1]
    min_x = bg_width-1
    max_x = 0
    min_y = bg_height-1
    max_y = 0
    ###text lines
    for src_quad_item in src_quad_list:
        #print('src_quad_list[0]:', src_quad_list[0])
        p1 = np.array([(bg_width - fg_width) / 2 + int(src_quad_item[0][0]*fg_width), (bg_height - fg_height) / 2 + int(src_quad_item[0][1]*fg_height), z_value, 0], np.float32) - pcenter
        p2 = np.array([(bg_width - fg_width) / 2 + int(src_quad_item[1][0]*fg_width), (bg_height - fg_height) / 2 + int(src_quad_item[1][1]*fg_height), z_value, 0], np.float32) - pcenter
        p3 = np.array([(bg_width - fg_width) / 2 + int(src_quad_item[2][0]*fg_width), (bg_height - fg_height) / 2 + int(src_quad_item[2][1]*fg_height), z_value, 0], np.float32) - pcenter
        p4 = np.array([(bg_width - fg_width) / 2 + int(src_quad_item[3][0]*fg_width), (bg_height - fg_height) / 2 + int(src_quad_item[3][1]*fg_height), z_value, 0], np.float32) - pcenter
        #exit()
        dst1 = r.dot(p1)
        dst2 = r.dot(p2)
        dst3 = r.dot(p3)
        dst4 = r.dot(p4)
        list_dst = [dst1, dst2, dst3, dst4]
        cont_dst = np.zeros((4, 2), np.float32)
        # projection transform
        dst_quad_item=[]
        for i in range(4):
            cont_dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
            cont_dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]
            if cont_dst[i, 0]<min_x:
                min_x=cont_dst[i, 0]
            if cont_dst[i, 0]>max_x:
                max_x=cont_dst[i, 0]
            if cont_dst[i, 1]<min_y:
                min_y=cont_dst[i, 0]
            if cont_dst[i, 1]>max_y:
                max_y=cont_dst[i, 0]
            dst_point_item=[]
            dst_point_item.append(int(cont_dst[i, 0]))
            dst_point_item.append(int(cont_dst[i, 1]))
            dst_quad_item.append(dst_point_item)
        dst_quad_list.append(dst_quad_item)

    if max_x<bg_width-1 and min_x>0 and min_y >0 and max_y < bg_height-1:
        try:
            warpR = cv2.getPerspectiveTransform(org, dst)
            im_out = cv2.warpPerspective(fg_img, warpR, (bg_img.shape[1],bg_img.shape[0]))# flags = cv2.INTER_NEAREST,borderMode=cv2.BORDER_TRANSPARENT)
        except  Exception as e:
            print('Exception while executing: getPerspectiveTransform&&warpPerspective')
            exit()
            return False, dst_quad_list, 0
        for i in range(bg_img.shape[1]):
            for j in range(bg_img.shape[0]):
                pixel = im_out[j, i]
                if not np.all(pixel == [0, 0, 0]):
                    bg_img[j, i] = im_out[j, i]
        color = (255, 255, 0)
        thickness = 2
        #cv2.line(bg_img, tuple([dst[0, 0], dst[0, 1]]), tuple([dst[1, 0], dst[1, 1]]), color, thickness)
        #cv2.line(bg_img, tuple([dst[1, 0], dst[1, 1]]), tuple([dst[2, 0], dst[2, 1]]), color, thickness)
        #cv2.line(bg_img, tuple([dst[2, 0], dst[2, 1]]), tuple([dst[3, 0], dst[3, 1]]), color, thickness)
        #cv2.line(bg_img, tuple([dst[3, 0], dst[3, 1]]), tuple([dst[0, 0], dst[0, 1]]), color, thickness)
        return True, dst_quad_list, bg_img
    else:
        print('warpped bbox out of bg_img, drop')
        #exit()
        #print('bad warpPerspective...drop...')
        #exit()
        return False, dst_quad_list, 0

def cv2_base64(image):
    base64_str = cv2.imencode('.jpg',image)[1].tostring()
    base64_str = base64.b64encode(base64_str)
    return base64_str

def process(src_img, bg_img, src_quad_list, label_list, dst_img_dir, dst_img_name):
    synth_flag = False
    while not synth_flag:
        synth_flag, dst_quad_list, synthed_img = synth_img(src_img, bg_img, src_quad_list)
    #for dst_quad_item in dst_quad_list:
        #color = (255, 0, 0)
        #thickness=2
        #cv2.line(synthed_img, tuple(dst_quad_item[0]), tuple(dst_quad_item[1]), color, thickness)
        #cv2.line(synthed_img, tuple(dst_quad_item[1]), tuple(dst_quad_item[2]), color, thickness)
        #cv2.line(synthed_img, tuple(dst_quad_item[2]), tuple(dst_quad_item[3]), color, thickness)
        #cv2.line(synthed_img, tuple(dst_quad_item[3]), tuple(dst_quad_item[0]), color, thickness)
    if not synth_flag:
        exit()
        print('synth_img error while processing:')
        return False, 0, 0
    else:
        pil_synthed_img = Image.fromarray(cv2.cvtColor(synthed_img, cv2.COLOR_BGR2RGB))
        pil_synthed_img.save(dst_img_dir+dst_img_name)
        img_h, img_w, channels = synthed_img.shape
        im_file = dst_img_dir + dst_img_name
        #with open(im_file, 'rb') as f:
            #image = f.read()
            #image_base64 = str(base64.b64encode(image), encoding='utf-8')
            #imageData = image_base64
        imageData = "image data"
        shapes=[]
        fillColor = [255, 0, 0, 128]
        lineColor = [0, 255, 0, 128]
        version = "4.2.9"
        for idx in range(len(dst_quad_list)):
            dst_quad_item = dst_quad_list[idx]
            label = label_list[idx]
            shapes.append(dict_shapes(dst_quad_item, label))
            json_data=dict_json(version, imageData, shapes, dst_img_name, img_h, img_w, fillColor, lineColor)
        return True, json_data, synthed_img