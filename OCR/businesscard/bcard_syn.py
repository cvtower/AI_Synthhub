# coding:utf-8
import os
import sys
#import PIL.Image as PImage
from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np
from faker import Faker
import random
from PIL import Image
from PIL import ImageEnhance, ImageFilter
from annotation_synth_inv import process
import json
import colorsys

fake_engine_cn = Faker(locale="zh_CN")
fake_engine_en = Faker(locale="en_US")

dst_img_dir = './dst_clean_version/'
dst_cnt = 1000000
cur_cnt = 0
target_w_list = [90,90,85,90,90,54,50,54,45,40]
target_h_list = [54,50,54,45,40,90,90,85,90,90]
scale_factor = 10
min_x_res = 0.1
min_y_res = 0.1
max_x_res = 0.15
max_y_res = 0.15

max_name_size = 50
min_name_size = 40
debug_flag = True
debug_l2 = False

def getallfile(path):
    allpath=[]
    allname=[]
    allfilelist=os.listdir(path)
    # 遍历该文件夹下的所有目录或者文件
    for file in allfilelist:
        filepath=os.path.join(path,file)
        # 如果是文件夹，递归调用函数
        if os.path.isdir(filepath):
            getallfile(filepath)
        # 如果不是文件夹，保存文件路径及文件名
        elif os.path.isfile(filepath):
            allpath.append(filepath)
            allname.append(file)
    return allpath, allname

def get_random_bg(img_list, img_dir, logo_loc_flag, target_cnt=1):
    img_item = random.sample(img_list, target_cnt)
    #print(img_item[0])
    img_path = img_dir+img_item[0]
    #target_size_idx = random.randint(0,len(target_w_list)-1)
    if logo_loc_flag ==0 or logo_loc_flag==1:
        target_size_idx = np.random.choice(np.arange(0, len(target_w_list)), p=[0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0])
    elif logo_loc_flag ==2:
        target_size_idx = np.random.choice(np.arange(0, len(target_w_list)), p=[0, 0, 0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2])
    target_w = int(scale_factor*target_w_list[target_size_idx])
    target_h = int(scale_factor*target_h_list[target_size_idx])
    bg_img = cv2.imread(img_path)
    target_size = (target_w, target_h)
    #print('bg_img_path:', img_path)
    #print('target_size:', target_size)
    try:
        target_bg_img = cv2.resize(bg_img, target_size)
    #print(target_bg_img.shape)
        return target_bg_img, True, img_path#, target_size_idx
    except  Exception as e:
        print('target_bg_img = cv2.resize exception:', img_path)
        exit
        return 0, False, img_path

def get_random_logo(img_list, img_dir, target_cnt=1):
    img_item = random.sample(img_list, target_cnt)
    #print('logo_img_path', img_item[0])
    img_path = img_dir+img_item[0]
    logo_img = cv2.imread(img_path)
    return logo_img#, target_size_idx

def gen_background(img_bg, img_logo, qr_img, logo_loc_flag):
    min_logo_scale = 0.15
    max_logo_scale = 0.3
    #print(img_bg.shape)
    bg_h, bg_w, channels = img_bg.shape
    #print(channels)
    #exit()
    ori_logo_h, ori_logo_w, channels = img_logo.shape
    pos_x = int(0.5*bg_w)-ori_logo_w//2
    logo_w = ori_logo_w
    logo_h = ori_logo_h
    if logo_loc_flag ==2:
        min_pos_y = min_y_res * bg_h
        max_pos_y = 0.4*bg_h - logo_h
        pos_y = int(random.uniform(min_pos_y, max_pos_y))
        logo_h = random.uniform(min_logo_scale, max_logo_scale) * bg_h
        logo_w = logo_h*(ori_logo_w/ori_logo_h)
        pos_x = int((bg_w-logo_w)/2)
    elif logo_loc_flag ==0 or logo_loc_flag ==1:
        logo_w = random.uniform(min_logo_scale, max_logo_scale) * bg_w
        logo_h = logo_w*(ori_logo_h/ori_logo_w)
        if logo_loc_flag ==0:
            #pos_x=int(random.uniform(min_x_res, max_x_res) * bg_w)
            pos_x = int(bg_w * min_x_res)
            pos_y = int(bg_h * min_y_res)
        if logo_loc_flag ==1:
            #pos_x=int(random.uniform(0.5+min_x_res, 0.5+max_x_res) * bg_w)
            pos_x = int(bg_w * 0.6)
            pos_y = int(bg_h * min_y_res)

    target_size = (int(logo_w), int(logo_h))
    img_logo_resized = cv2.resize(img_logo, target_size)
    qr_img_resized = cv2.resize(qr_img, (int(logo_w),int(logo_w)))
    rows, cols, channels = img_logo_resized.shape

    if pos_x<0 or pos_y<0:
        return img_logo_resized, 1, 0, 0
    if logo_loc_flag == 0:
        if logo_w+pos_x>=(bg_w-1)//2 or logo_h+pos_y >(bg_h-1):
            return img_logo_resized, 1, 0, 0
    elif logo_loc_flag == 1:
        if logo_h+pos_y >=(bg_h-1):
            return img_logo_resized, 1, 0, 0
    #print(pos_y,rows, pos_x,cols)
    roi = img_bg[pos_y:rows+pos_y, pos_x:cols+pos_x]

    img2gray = cv2.cvtColor(img_logo_resized, cv2.COLOR_BGR2GRAY)
    img2gray = cv2.medianBlur(img2gray, 5)
    ret, mask = cv2.threshold(img2gray, 127, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    try:
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        img2_fg = cv2.bitwise_and(img_logo_resized, img_logo_resized, mask=mask)

        dst = cv2.add(img1_bg, img2_fg)
        img_bg[pos_y:rows+pos_y, pos_x:cols+pos_x] = dst
        #qrcode
        qr_rows, qr_cols, qr_channels = qr_img_resized.shape
        qr_pos_y = bg_h-pos_y-qr_rows
        img_bg[qr_pos_y:qr_rows + qr_pos_y, pos_x:qr_cols + pos_x] = qr_img_resized
        if logo_loc_flag == 0:
            return img_bg, 0, cols+pos_x, rows//2+pos_y
        elif logo_loc_flag == 1:
            return img_bg, 0, int(bg_w*min_x_res), rows//2+pos_y
        elif logo_loc_flag == 2:
            #print(pos_x, pos_y)
            return img_bg, 0, pos_x, pos_y+logo_h
    except Exception as e:
        return img_logo_resized, 1, 0, 0

def get_random_font(font_list, target_cnt=1):
    selected_font = random.sample(font_list, target_cnt)
    return selected_font[0]


def augument(image):

    min_value = 0.5
    max_value = 1.5
    # 亮度
    enh_bri = ImageEnhance.Brightness(image)
    brightness = random.uniform(min_value, max_value)
    image_brightened = enh_bri.enhance(brightness)

    # 色度,增强因子为1.0是原始图像
    # 色度
    enh_col = ImageEnhance.Color(image_brightened)
    min_value = 0.6
    max_value = 1.4
    color = random.uniform(min_value, max_value)
    image_colored = enh_col.enhance(color)

    # 对比度，增强因子为1.0是原始图片
    enh_con = ImageEnhance.Contrast(image_colored)
    min_value = 0.8
    max_value = 1.3
    contrast = random.uniform(min_value, max_value)
    image_contrasted = enh_con.enhance(contrast)

    # 锐度，增强因子为1.0是原始图片
    enh_sha = ImageEnhance.Sharpness(image_contrasted)
    min_value = 0.8
    max_value = 1.5
    sharpness = random.uniform(min_value, max_value)
    image_sharped = enh_sha.enhance(sharpness)

    min_value = 1
    max_value = 2
    radius = random.randint(min_value, max_value)
    image_blurred = image_sharped.filter(ImageFilter.GaussianBlur(radius=radius))
    return image_blurred

def quad_store(text_x, text_offset_x, text_y, text_offset_y, text_w, text_h, img_width, img_height, half_post_prefix_x, half_post_prefix_y):
    quad_item=[]
    vertex_item = []
    vertex_item.append(int(text_x-half_post_prefix_x)/img_width)
    vertex_item.append(int(text_y-half_post_prefix_y)/img_height)
    quad_item.append(vertex_item)
    vertex_item = []
    vertex_item.append(int(text_x+text_w+half_post_prefix_x)/img_width)
    vertex_item.append(int(text_y-half_post_prefix_y)/img_height)
    quad_item.append(vertex_item)
    vertex_item = []
    vertex_item.append(int(text_x+text_w+half_post_prefix_x)/img_width)
    vertex_item.append(int(text_y+text_h+half_post_prefix_y)/img_height)
    quad_item.append(vertex_item)
    vertex_item = []
    vertex_item.append(int(text_x-half_post_prefix_x)/img_width)
    vertex_item.append(int(text_y+text_h+half_post_prefix_y)/img_height)
    quad_item.append(vertex_item)
    return quad_item

def compute_inv_color(img):
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []
    per_image_Bmean.append(np.mean(img[:,:,0]))
    per_image_Gmean.append(np.mean(img[:,:,1]))
    per_image_Rmean.append(np.mean(img[:,:,2]))

    R_mean = int(np.mean(per_image_Rmean))
    G_mean = int(np.mean(per_image_Gmean))
    B_mean = int(np.mean(per_image_Bmean))
    if (R_mean * 0.299 + G_mean * 0.587 + B_mean * 0.114) > 128:
        R_mean = random.randint(0, 15)
        G_mean = random.randint(0, 15)
        B_mean = random.randint(0, 15)
    else:
        R_mean = random.randint(255-15, 255)
        G_mean = random.randint(255-15, 255)
        B_mean = random.randint(255-15, 255)
    return B_mean, G_mean, R_mean


if __name__ == '__main__':

    cn_font_path = []
    cn_font_name = []
    en_font_dir = './en_font/'
    cn_font_dir = './cn_font/'
    bg_img_dir = './bcard_bg_total/'
    logo_img_dir = './LLD-logo_sample/'
    qr_img_dir = './qr_samples/'
    sbg_img_dir = './sbg_img/'
    coco_img_dir = './total_bg/'
    post_prefix = 'H'
    en_flag = 0
    logo_loc_flag = 0
    target_coco_cnt = 1
    cn_list = []
    en_list = []
    en_files, en_names = getallfile(en_font_dir)
    cn_files, cn_names = getallfile(cn_font_dir)
    bg_img_list = os.listdir(bg_img_dir)
    logo_img_list = os.listdir(logo_img_dir)
    qr_img_list = os.listdir(qr_img_dir)
    coco_img_list = os.listdir(coco_img_dir)

    for dst_idx in range(dst_cnt):
        Faker.seed(dst_idx)
        quad_list = []
        label_list = []
        if (dst_idx % 4)==0:
            en_flag = 1
        logo_loc_flag = cur_cnt % 10#ud/lr/rl
        if logo_loc_flag == 0:
            logo_loc_flag=2
        elif logo_loc_flag%2 ==0:
            logo_loc_flag=1
        else:
            logo_loc_flag = 0
        rand_flag = False
        bg_img, rand_flag, bg_img_name = get_random_bg(bg_img_list, bg_img_dir, logo_loc_flag)
        if rand_flag == False:
            continue
        logo_img, rand_flag, logo_img_name = get_random_bg(logo_img_list, logo_img_dir, logo_loc_flag)
        if rand_flag == False:
            continue
        qr_img, rand_flag, qr_img_name = get_random_bg(qr_img_list, qr_img_dir, logo_loc_flag)
        if rand_flag == False:
            continue
        coco_img_path = coco_img_dir + str(random.sample(coco_img_list, target_coco_cnt)[0])
        sbg_img, status_flag, bg_x, bg_y = gen_background(bg_img, logo_img, qr_img, logo_loc_flag)
        print('coco_img loc:', coco_img_path)
        coco_img  = cv2.imread(coco_img_path)
        bg_pil_img = Image.fromarray(cv2.cvtColor(sbg_img, cv2.COLOR_BGR2RGB))
        r, g, b = compute_inv_color(bg_img)
        if status_flag == 0:
            sbg_img_name = sbg_img_dir + str(dst_idx) + '.jpg'
            #cv2.imwrite(sbg_img_name, sbg_img)
            en_font = get_random_font(en_files)
            cn_font = get_random_font(cn_files)
            #print('cn_font:',cn_font)

            cn_company = fake_engine_cn.company()
            cn_name = fake_engine_cn.name()
            name_prefix_flag = False
            name_prefix = '姓名：'
            cn_name_list = list(cn_name)
            if len(cn_name)==2:
                if (random.uniform(0, 1)) < 0.3:
                    cn_name = cn_name[0]+' '+ cn_name[1]
                elif (random.uniform(0, 1)) > 0.7:
                    cn_name = cn_name[0] + '  ' + cn_name[1]
                else:
                    cn_name = cn_name[0] + '   ' + cn_name[1]
            elif len(cn_name)==3:
                if (random.uniform(0, 1)) < 0.7:
                    cn_name = cn_name[0]+' '+ cn_name[1]+' '+ cn_name[2]
                else:
                    cn_name = cn_name[0]+'  '+ cn_name[1]+'  '+ cn_name[2]
            spacing_len = random.randint(1, 3)
            if (random.uniform(0, 1))<0.05:
                name_prefix_flag = True
                cn_name = name_prefix + cn_name

            en_name = fake_engine_en.name()
            cn_job = fake_engine_cn.job()
            en_job = fake_engine_en.job()
            cn_address = fake_engine_cn.address().split(' ')[0]
            cn_phone = fake_engine_cn.phone_number()
            cn_email = fake_engine_cn.email()
            temp_rand = random.uniform(0, 1)
            if temp_rand > 0.7:
                fix_tel = '0'+ str(cn_phone)[1:3] + '-' + str(str(random.randint(0,99999999)).zfill(8))
            elif temp_rand<0.3:
                fix_tel = '0'+ str(cn_phone)[1:3] + '  ' + str(str(random.randint(0,99999999)).zfill(8))
            else:
                fix_tel = '(0'+ str(cn_phone)[1:3] + ')' + str(str(random.randint(0,99999999)).zfill(8))
            #if debug_flag:
                #print(cn_name, cn_job, en_job, cn_company, cn_address, cn_phone, cn_email, fix_tel)
                #exit()
            name_size = random.randint(min_name_size, max_name_size)
            res_font_size = random.randint(10, 20)
            company_size = name_size + res_font_size
            content_size = name_size - res_font_size
            res_font_size = random.randint(10, 20)
            name_size = name_size + res_font_size

            selected_company_font = ImageFont.truetype(cn_font, company_size)
            selected_name_font = ImageFont.truetype(cn_font, name_size)
            selected_res_font = ImageFont.truetype(cn_font, content_size)
            draw = ImageDraw.Draw(bg_pil_img)

            #print(cn_name)
            cn_company_offset_x, cn_company_offset_y = selected_company_font.getoffset(cn_company)
            en_name_offset_x, en_name_offset_y = selected_name_font.getoffset(en_name)
            cn_name_offset_x, cn_name_offset_y = selected_name_font.getoffset(cn_name)
            cn_address_offset_x, cn_address_offset_y = selected_res_font.getoffset(cn_address)
            fix_tel_offset_x, fix_tel_offset_y = selected_res_font.getoffset(fix_tel)
            cn_phone_offset_x, cn_phone_offset_y = selected_res_font.getoffset(cn_phone)
            width, height = bg_pil_img.size
            if logo_loc_flag==0 or logo_loc_flag==1 or logo_loc_flag==2:
                #company name
                address_w, address_h = draw.textsize(cn_address, font=selected_res_font)
                company_w, company_h = draw.textsize(cn_company, font=selected_company_font)
                if logo_loc_flag ==2:
                   if address_w>=int(width*0.9-1) or company_w>=int(width*0.9-1):
                       continue
                   if address_w>company_w:
                      bg_x = (width-address_w)//2
                   else:
                      bg_x = (width-company_w)//2
                cn_company_x = bg_x
                cn_company_y = bg_y-company_h//2
                if logo_loc_flag == 2:
                    max_char_w, max_char_h = draw.textsize(cn_address, font=selected_res_font)
                    cn_company_x = (width-company_w)//2
                if logo_loc_flag == 1:
                    if cn_company_x+company_w>=int(width*0.6):
                        continue
                if cn_company_x+company_w>=(width-1):
                    continue
                draw.text((cn_company_x, cn_company_y), cn_company, fill=(r, g, b, 255), font=selected_company_font)
                post_prefix_x, post_prefix_y = draw.textsize(post_prefix, font=selected_company_font)
                quad_item = quad_store(cn_company_x, cn_company_offset_x, cn_company_y, cn_company_offset_y, company_w, company_h, width, height, post_prefix_x//2, post_prefix_y/8)
                quad_list.append(quad_item)
                label_list.append(cn_company)

                bi_tel_flag = 1 if (random.uniform(0, 1))<0.5 else 0
                #name
                text_w, text_h = draw.textsize(cn_name, font=selected_name_font)
                res_y = (int(height*(1-min_y_res))-bg_y-text_h)
                if res_y<=4*text_h or (bg_x+text_w)>=(1-min_x_res)*width:
                    continue
                cn_name_x = bg_x
                cn_name_y = cn_company_y+company_h//2+res_y//2-int(1.5*int(text_h))
                post_prefix_x, post_prefix_y = draw.textsize(post_prefix, font=selected_name_font)
                quad_item = quad_store(cn_name_x, cn_name_offset_x, cn_name_y, cn_name_offset_y, text_w, text_h, width, height, post_prefix_x//2, post_prefix_y/8)
                quad_list.append(quad_item)
                label_list.append(cn_name)
                draw.text((cn_name_x, cn_name_y), cn_name, fill=(r, g, b, 255), font=selected_name_font, spacing=name_size*spacing_len)

                #phone
                if (random.uniform(0, 1)) < 0.4:
                    cn_phone = ('电话: ' + cn_phone)
                    fix_tel = ('电话: ' + fix_tel)
                text_w, text_h = draw.textsize(cn_phone, font=selected_res_font)
                cn_phone_x = bg_x
                cn_phone_y = cn_company_y+company_h//2+res_y//2
                post_prefix_x, post_prefix_y = draw.textsize(post_prefix, font=selected_res_font)
                quad_item = quad_store(cn_phone_x, cn_phone_offset_x, cn_phone_y, cn_phone_offset_y, text_w, text_h, width, height, post_prefix_x//2, post_prefix_y/8)
                quad_list.append(quad_item)
                label_list.append(cn_phone)
                draw.text((cn_phone_x, cn_phone_y), cn_phone, fill=(r, g, b, 255), font=selected_res_font)
                if bi_tel_flag:
                    text_w, text_h = draw.textsize(fix_tel, font=selected_res_font)
                    fix_tel_x = bg_x
                    fix_tel_y = cn_company_y + company_h // 2 + res_y // 2+int(text_h)
                    quad_item = quad_store(fix_tel_x, fix_tel_offset_x, fix_tel_y, fix_tel_offset_y, text_w, text_h, width, height, post_prefix_x//2, post_prefix_y/8)
                    quad_list.append(quad_item)
                    label_list.append(fix_tel)
                    draw.text((fix_tel_x, fix_tel_y), fix_tel, fill=(r, g, b, 255), font=selected_res_font)
                    #address
                    text_w, text_h = draw.textsize(cn_address, font=selected_res_font)
                    cn_address_x = bg_x
                    cn_address_y = cn_company_y+company_h//2+res_y//2+2*int(text_h)
                    quad_item = quad_store(cn_address_x, cn_address_offset_x, cn_address_y, cn_address_offset_y, text_w, text_h, width, height, post_prefix_x//2, post_prefix_y/8)
                    quad_list.append(quad_item)
                    label_list.append(cn_address)
                    draw.text((cn_address_x, cn_address_y), cn_address, fill=(r, g, b, 255), font=selected_res_font)
                else:
                    text_w, text_h = draw.textsize(cn_address, font=selected_res_font)
                    cn_address_x = bg_x
                    cn_address_y = cn_company_y+company_h//2+res_y//2+int(text_h)
                    quad_item = quad_store(cn_address_x, cn_address_offset_x, cn_address_y, cn_address_offset_y, text_w, text_h, width, height, post_prefix_x//2, post_prefix_y/8)
                    quad_list.append(quad_item)
                    label_list.append(cn_address)
                    draw.text((cn_address_x, cn_address_y), cn_address, fill=(r, g, b, 255), font=selected_res_font)

                if logo_loc_flag ==0 and (bg_x + text_w) >= (1 - min_x_res) * width:
                    continue
                if logo_loc_flag == 1 and (bg_x + text_w) >=  width//2:
                    continue
                auged_img = augument(bg_pil_img)
                color = (255, 0, 0)
                thickness = 2
                sbg_img = np.array(auged_img)
                if debug_l2:
                    for quad_item in quad_list:
                        cv2.line(sbg_img, tuple([int(quad_item[0][0]*width),int(quad_item[0][1]*height)]), tuple([int(quad_item[1][0]*width),int(quad_item[1][1]*height)]), color, thickness)
                        cv2.line(sbg_img, tuple([int(quad_item[1][0]*width),int(quad_item[1][1]*height)]), tuple([int(quad_item[2][0]*width),int(quad_item[2][1]*height)]), color, thickness)
                        cv2.line(sbg_img, tuple([int(quad_item[2][0]*width),int(quad_item[2][1]*height)]), tuple([int(quad_item[3][0]*width),int(quad_item[3][1]*height)]), color, thickness)
                        cv2.line(sbg_img, tuple([int(quad_item[3][0]*width),int(quad_item[3][1]*height)]), tuple([int(quad_item[0][0]*width),int(quad_item[0][1]*height)]), color, thickness)
                #bg_pil_img = Image.fromarray(cv2.cvtColor(sbg_img, cv2.COLOR_BGR2RGB))
                dst_img_name = str(cur_cnt) + '.jpg'
                #dst_img_name = dst_img_dir + dst_img_name
                succeed_flag = False
                while not succeed_flag:
                    print(bg_img_name, logo_img_name, qr_img_name)
                    succeed_flag, json_data, synthed_img = process(sbg_img, coco_img, quad_list, label_list, dst_img_dir, dst_img_name)
                dst_json_name = dst_img_dir+str(cur_cnt) + '.json'
                #cv2.write(dst_img_name, synthed_img)
                #pil_synthed_img = Image.fromarray(cv2.cvtColor(synthed_img, cv2.COLOR_BGR2RGB))
                #pil_synthed_img.save(dst_img_name)
                json.dump(json_data, open(dst_json_name, 'w'))
                cur_cnt=cur_cnt+1
                if cur_cnt==20000:
                    print('synth finished')
                    exit()
        else:
            print('gen_background failed...continue')
            continue
