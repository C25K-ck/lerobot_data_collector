
import h5py
import cv2
import os
import re

import rclpy
import time
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys

sys.path.append("/home/dreame/test/moca-metap/test_offline_data/")
from lcm_unit import lcmUnit
from ros_node import *


def extract_number(filename):
    # 使用正则表达式找到文件名中的第一个数字序列并返回
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group(0))
    else:
        return float('inf')  

class checkHDF5(offlineControlRos):
    def __init__(self) -> None:
        offlineControlRos.__init__(self)
        self.lcm_unit = lcmUnit()
        self.data_it = self.data_iter()
        time.sleep(1)



    def data_iter(self):

        while True:
            for i in range(1,2):

                self.start_save_data(0)

                hdf5_path = "/home/lifeng/data/act/task_1/"
                hdf5_file_name = f"episode_qpos_{i}.hdf5"
                with h5py.File(os.path.join(hdf5_path, hdf5_file_name), 'r') as root:
                    head_list = root[f'/observations/images/head']
                    left_hand_list = root[f'/observations/images/left_hand']
                    right_hand_list = root[f'/observations/images/right_hand']
                    qpos_list = root[f'/observations/qpos']
                    action_list = root[f'/action']

                    for i in range(len(head_list)):

                        head_rgb = head_list[i]
                        right_hand_rgb = right_hand_list[i]
                        left_hand_rgb = left_hand_list[i]
                        head_bgr = cv2.cvtColor(head_rgb, cv2.COLOR_RGB2BGR)
                        left_hand_bgr = cv2.cvtColor(left_hand_rgb, cv2.COLOR_RGB2BGR)
                        right_hand_bgr = cv2.cvtColor(right_hand_rgb, cv2.COLOR_RGB2BGR)
                        action = action_list[i]
                        qpos = qpos_list[i]
                        print(action.tolist())

                        yield action

                        print(f"shape : 1{head_bgr.shape}\n")
                        print(f"shape : 2{ self.image_list[0].shape}\n")

                        head_img = cv2.addWeighted(head_bgr, 0.6, self.image_list[0], 0.4, 1)

                        right_hand_img = cv2.addWeighted(right_hand_bgr, 0.6, self.image_list[2], 0.8, 0)


                        cv2.imshow('head', head_img)

                        cv2.imshow('left_hand', left_hand_bgr)
                        cv2.imshow('right_hand', right_hand_img)
                        cv2.waitKey(1)
                        

                    




    def period_get_data(self):
        
        action = next(self.data_it)
        if False:
            self.lcm_unit.send_to_robot(np.array(action))
        
        self.pub_act_qpos(action)
        pass
    

if __name__ =="__main__":
    
    rclpy.init()
    robot_ctrl_ros = checkHDF5()

        # 运行节点
    rclpy.spin(robot_ctrl_ros)
    #
    # # 销毁节点，退出ROS2
    robot_ctrl_ros.destroy_node()
    rclpy.shutdown()

