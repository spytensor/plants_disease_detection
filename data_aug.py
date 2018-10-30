from PIL import Image,ImageEnhance,ImageFilter,ImageOps
import os
import shutil
import numpy as np
import cv2
import random
from skimage.util import random_noise
from skimage import exposure


image_number = 0

raw_path = "./data/train/"

new_path = "./aug/train/"

# 加高斯噪声
def addNoise(img):
    '''
    注意：输出的像素是[0,1]之间,所以乘以5得到[0,255]之间
    '''
    return random_noise(img, mode='gaussian', seed=13, clip=True)*255

def changeLight(img):
    rate = random.uniform(0.5, 1.5)
    # print(rate)
    img = exposure.adjust_gamma(img, rate) #大于1为调暗，小于1为调亮;1.05
    return img

try:
    for i in range(59):
        os.makedirs(new_path + os.sep + str(i))
    except:
        pass

for raw_dir_name in range(59):

    raw_dir_name = str(raw_dir_name)

    saved_image_path = new_path + raw_dir_name+"/"

    raw_image_path = raw_path + raw_dir_name+"/"

    if not os.path.exists(saved_image_path):

        os.mkdir(saved_image_path)

    raw_image_file_name = os.listdir(raw_image_path)

    raw_image_file_path = []

    for i in raw_image_file_name:

        raw_image_file_path.append(raw_image_path+i)

    for x in raw_image_file_path:

        img = Image.open(x)
        cv_image = cv2.imread(x)

        # 高斯噪声
        gau_image = addNoise(cv_image)
        # 随机改变
        light = changeLight(cv_image)
        light_and_gau = addNoise(light)

        cv2.imwrite(saved_image_path + "gau_" + os.path.basename(x),gau_image)
        cv2.imwrite(saved_image_path + "light_" + os.path.basename(x),light)
        cv2.imwrite(saved_image_path + "gau_light" + os.path.basename(x),light_and_gau)
        #img = img.resize((800,600))

        #1.翻转 

        img_flip_left_right = img.transpose(Image.FLIP_LEFT_RIGHT)

        img_flip_top_bottom = img.transpose(Image.FLIP_TOP_BOTTOM)

        #2.旋转 

        #img_rotate_90 = img.transpose(Image.ROTATE_90)

        #img_rotate_180 = img.transpose(Image.ROTATE_180)

        #img_rotate_270 = img.transpose(Image.ROTATE_270)

        #img_rotate_90_left = img_flip_left_right.transpose(Image.ROTATE_90)

        #img_rotate_270_left = img_flip_left_right.transpose(Image.ROTATE_270)

        #3.亮度

        #enh_bri = ImageEnhance.Brightness(img)
        #brightness = 1.5
        #image_brightened = enh_bri.enhance(brightness)

        #4.色彩

        #enh_col = ImageEnhance.Color(img)
        #color = 1.5

        #image_colored = enh_col.enhance(color)

        #5.对比度

        enh_con = ImageEnhance.Contrast(img)

        contrast = 1.5

        image_contrasted = enh_con.enhance(contrast)

        #6.锐度

        #enh_sha = ImageEnhance.Sharpness(img)
        #sharpness = 3.0

        #image_sharped = enh_sha.enhance(sharpness)

        #保存 

        img.save(saved_image_path + os.path.basename(x))

        img_flip_left_right.save(saved_image_path + "left_right_" + os.path.basename(x))

        img_flip_top_bottom.save(saved_image_path + "top_bottom_" + os.path.basename(x))

        #img_rotate_90.save(saved_image_path + "rotate_90_" + os.path.basename(x))

        #img_rotate_180.save(saved_image_path + "rotate_180_" + os.path.basename(x))

        #img_rotate_270.save(saved_image_path + "rotate_270_" + os.path.basename(x))

        #img_rotate_90_left.save(saved_image_path + "rotate_90_left_" + os.path.basename(x))

        #img_rotate_270_left.save(saved_image_path + "rotate_270_left_" + os.path.basename(x))

        #image_brightened.save(saved_image_path + "brighted_" + os.path.basename(x))

        #image_colored.save(saved_image_path + "colored_" + os.path.basename(x))

        image_contrasted.save(saved_image_path + "contrasted_" + os.path.basename(x))

        #image_sharped.save(saved_image_path + "sharped_" + os.path.basename(x))

        image_number += 1

        print("convert pictur" "es :%s size:%s mode:%s" % (image_number, img.size, img.mode))

 
