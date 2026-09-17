import h5py
import cv2
import os
import re
import numpy as np
import time

import random

from conf import *

MASK_FILE_LIST = list(filter(lambda x: x.endswith(".jpg"), os.listdir("/home/dreame/data/hf_dataset")))


def generate_green_screen_mask_image_head(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
 
    # 步骤3: 创建绿色范围的掩码
    # 绿色在HSV中的范围大致为：Hue(100-140), Saturation(50-255), Value(50-255)
    lower_green = np.array([40, 100+random.randrange(-5,5), 60+random.randrange(-5,5)])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    return mask
 

def generate_green_screen_mask_image_wrist(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
 
    # 步骤3: 创建绿色范围的掩码
    # 绿色在HSV中的范围大致为：Hue(100-140), Saturation(50-255), Value(50-255)

    lower_green = np.array([30, 160+random.randrange(-5,5), 80+random.randrange(-20,10)])
    upper_green = np.array([50, 255, 255])

    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    return mask


def generate_mask(image,mask_threshold,noise):

    low_rand = [random.randrange(-noise,noise), random.randrange(-noise,noise), random.randrange(-noise,noise)]
    low_bound = np.array([mask_threshold[0]+low_rand[0],mask_threshold[1]+low_rand[1],mask_threshold[2]+low_rand[2]])
    upper_rand = [random.randrange(-noise,noise), random.randrange(-noise,noise), random.randrange(-noise,noise)]
    upper_bound = np.array([mask_threshold[3]+upper_rand[0],mask_threshold[4]+upper_rand[1],mask_threshold[5]+upper_rand[2]])
    mask = cv2.inRange(image, low_bound, upper_bound)
    return mask




def load_mask_image():
    file_name = random.choice(MASK_FILE_LIST)
    file_path = "/data2/val2017/" + file_name
    image = cv2.imread(file_path)
    image = cv2.resize(image, (640,480))

    choice_list = [0, 1]
    val = random.choice(choice_list)
    if(val==1):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    
    return image

def add_random_lighting(image):
    alpha = 1 + np.random.uniform(-0.5, 0.5)  # 对比度变化因子
    beta = np.random.randint(-100, 100)         # 亮度变化值
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
 
    mean = 0
    sigma = np.random.uniform(0.2, 0.3)  # 噪声的标准差，模拟光照不均程度
    gauss = sigma * np.random.randn(image.shape[0], image.shape[1], 3) + mean  # 生成高斯噪声数组
    gauss = gauss.reshape(image.shape[0], image.shape[1], 3)  # 重塑为与原图相同的尺寸
    gauss = gauss.astype('uint8')  # 转换为uint8以便与原图相加
    # print(gauss)
    
    new_image = cv2.add(image, gauss)  # 将噪声添加到原图上
 
    return new_image
    

def merge_image(img2, img1, mask):
    img1_bg = cv2.bitwise_and(img1,img1,mask=mask)

    mask_inv = cv2.bitwise_not(mask)
    img2_bg = cv2.bitwise_and(img2,img2,mask=mask_inv)
    dst = cv2.add(img1_bg,img2_bg)
    return dst
