
import h5py
import cv2
import os
import re

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from conf import *
import numpy as np

def extract_number(filename):
    # 使用正则表达式找到文件名中的第一个数字序列并返回
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group(0))
    else:
        return float('inf')  

class addQvelUnit():
    def __init__(self) -> None:
        self.file_list = []
        self.load_files()
        self.iteration = self.get_data()

    def load_files(self):
        
        hdf5_path = (ACT_DATA_SAVE_PATH + TASK_NAME +"/")
        hdf5_number = os.listdir(hdf5_path)
        hdf5_files = sorted(hdf5_number, key=extract_number)
        for hdf5_file_name in hdf5_files:
            self.file_list.append(os.path.join(hdf5_path, hdf5_file_name))

    def get_data(self):
        head_list = []
        left_hand_list = []
        right_hand_list = []
        qpos_list = []
        action_list = []
    
                
        for hdf5_file_name in self.file_list:
            print(hdf5_file_name)
            with h5py.File(os.path.join(hdf5_file_name), 'r') as root:
                head_list = list(root[f'/observations/images/head'])
                left_hand_list = list(root[f'/observations/images/left_hand'])
                right_hand_list = list(root[f'/observations/images/right_hand'])
                qpos_list = list(root[f'/observations/qpos'])
                action_list = list(root[f'/action'])

                self.max_step = len(qpos_list)
                self.act_num = qpos_list[0].shape[0]

                action_vel_list = [np.zeros(self.act_num) for _ in range(self.max_step)] 
                for i in range(len(action_list)-1):
                    for j in range(self.act_num):
                        action_vel_list[i+1][j] = action_list[i+1][j]-action_list[i][j]
                print(f"head : {list(head_list)}\n")
            self.save_hdf5(hdf5_file_name, head_list, left_hand_list, right_hand_list, qpos_list, action_list, action_vel_list)

    def save_hdf5(self, file_name, head_list, left_hand_list, right_hand_list, qpos_list, action_list, action_vel_list):

        with h5py.File(file_name, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = False
            obs = root.create_group('observations')
            qpos = obs.create_dataset('qpos', (self.max_step, self.act_num))
            _ = root.create_dataset('action', (self.max_step, self.act_num))
            _ = root.create_dataset('action_vel', (self.max_step, self.act_num))
            image = obs.create_group('images')
            for cam_name in ["head", "left_hand", "right_hand"]:
                _ = image.create_dataset(cam_name, (self.max_step, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )

            root['/observations/images/head'][...] = head_list
            root['/observations/images/left_hand'][...] = left_hand_list
            root['/observations/images/right_hand'][...] = right_hand_list
            root['/observations/qpos'][...] = qpos_list
            root[f'/action'][...] = action_list
            root[f'/action_vel'][...] = action_vel_list

        print(f"save finish : {file_name}\n")

if __name__ =="__main__":
    add_qvel_manage = addQvelUnit()