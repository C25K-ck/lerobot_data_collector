
import h5py
import cv2
import os
import re

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from conf import *


def extract_number(filename):
    # 使用正则表达式找到文件名中的第一个数字序列并返回
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group(0))
    else:
        return float('inf')  

hdf5_path = (ACT_DATA_SAVE_PATH + TASK_NAME +"/")

hdf5_path = "/home/dreame/data/task_99"
hdf5_number = os.listdir(hdf5_path)
hdf5_files = sorted(hdf5_number, key=extract_number)
print(hdf5_files)


for hdf5_file_name in hdf5_files:
    print(hdf5_file_name)

    with h5py.File(os.path.join(hdf5_path, hdf5_file_name), 'r') as root:
        head_list = root[f'/observations/images/headf']
        # left_hand_list = root[f'/observations/images/left_hand']
        right_hand_list = root[f'/observations/images/right_hand']
        qpos_list = root[f'/observations/qpos']
        action_list = root[f'/action']


        count = 0
        for i in range(len(head_list)):
            print(i)
            if i % 5 == 0:
                head_rgb = head_list[i]
                right_hand_rgb = right_hand_list[i]
                # left_hand_rgb = left_hand_list[i]
                head_bgr = cv2.cvtColor(head_rgb, cv2.COLOR_RGB2BGR)
                # left_hand_bgr = cv2.cvtColor(left_hand_rgb, cv2.COLOR_RGB2BGR)
                right_hand_bgr = cv2.cvtColor(right_hand_rgb, cv2.COLOR_RGB2BGR)
                action = action_list[i]
                qpos = qpos_list[i]
                print(action.tolist())
                import sys
                sys.exit()
                print(qpos)

                # cv2.namedWindow("head", cv2.WINDOW_NORMAL)
                # cv2.namedWindow("right_hand", cv2.WINDOW_NORMAL)
                print(f"head_bgr : {head_bgr.shape}")

                print(f"right_hand_bgr : {right_hand_bgr.shape}")


                cv2.imshow('head', head_bgr)

                # cv2.imshow('left_hand', left_hand_bgr)
                cv2.imshow('right_hand', right_hand_bgr)
                cv2.waitKey(0)
