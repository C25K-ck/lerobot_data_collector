
# !/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("/home/dreame/test/factory_lerobot/act_eval")
import os
import copy
# 获取当前工作目录
pwd_path = os.getcwd()
# 往前后退 2 级
moca_path = os.path.dirname(os.path.dirname(pwd_path))
sys.path.append(moca_path)

from lcm import LCM

import numpy as np
import threading
from biped_lcm_types.python.upper_body_cmd_package import upper_body_cmd_package
from biped_lcm_types.python.upper_body_data_package import upper_body_data_package


class ArmState(object):
    def __init__(self):
        self.dimension = 30
        # self.default_arm_control_mode  = [4, 4, 4, 4, 4, 4, 4] + [4, 4, 4, 4, 4, 4, 4]
        self.default_arm_control_mode   = [200 for i in range(14)]
        self.default_hand_control_mode  = [200 for dim0 in range(12)]
        self.default_waist_control_mode = [4 for dim0 in range(2) ]
        self.default_head_control_mode = [4 for dim0 in range(2) ]
        self.default_control_mode       = self.default_arm_control_mode + self.default_hand_control_mode + self.default_waist_control_mode + self.default_head_control_mode

        self.is_used        = np.zeros(self.dimension, dtype= int)
        self.error_code     = np.zeros(self.dimension, dtype= int)
        self.status         = np.zeros(self.dimension, dtype= int)
        self.q              = np.zeros(self.dimension, dtype= np.float64)
        self.dot_q          = np.zeros(self.dimension, dtype= np.float64)
        self.current_or_torque = np.zeros(self.dimension, dtype= np.float64)

class lcmUnit(LCM):
    def __init__(self) -> None:
        # super().__init__('udpm://239.255.76.67:7667?ttl=1')
        super().__init__('udpm://239.255.76.67:7667?ttl=1')
        self.upper_body_cmd_topic   = 'upper_body_cmd'
        self.upper_body_data_topic  = 'upper_body_data'
        self.current_robot_state = ArmState()
        self.update_once_arm = False
        self.pre_pose = None
        self.speed_plan_pre_position = None
        self.speed_plan_pre_speed = None
        self.speed_plan_pre_acc = None
        self.interpolation_period = 2  # 单位是ms

        self.init_gripper_cnt = 0
        self.gripper_selection = [1, 1] # 1 gripper 0 hand

        self.subscribe(self.upper_body_data_topic, self.upper_body_data_listener_cb)
        self.pre_pose = np.zeros(30)
        self.lcm_thread_handle      = threading.Thread(target=self.lcm_handle, daemon=True)
        self.lcm_thread_handle.start()

    def upper_body_data_listener_cb(self, channel, data):
        try:
            msg = upper_body_data_package.decode(data)
            self.current_robot_state.q = np.array(msg.curJointPosVec)
            self.current_robot_state.status = np.array(msg.curStatusVec)
            self.current_robot_state.dot_q = np.array(msg.curSpeedVec)
            self.current_robot_state.current_or_torque = np.array(msg.curCurrentVec)
            self.update_once_arm = True
        except Exception as e:
            print(f"upper_body_data_listener_cb err: {e}")

    def lcm_handle(self):
        while True:
            self.handle()


    def send_to_robot(self, data_send, waist_val):
        data = np.array(copy.deepcopy(data_send))
        # print(data)
        data[:7] = np.array([0.5, 0.8, 1.9, -2, -2.5, 0.52, 0.0])
        # data[:7] = np.array([0.5, 0.8, 1.5, -2, -2.5, 0.52, 0.0])
        data[27] = 0
        data[26] = waist_val[0]
        data[20] *= 80



        upper_body_cmd_msg = self.load_upper_body_cmd_package(data)
        self.publish(self.upper_body_cmd_topic, upper_body_cmd_msg.encode())
        # print(f"send data: {self.upper_body_cmd_topic, upper_body_cmd_msg}\n")

    def load_upper_body_cmd_package(self, robot_30dof_solution):
        upper_body_cmd_msg  = upper_body_cmd_package()
        upper_body_cmd_msg.isUsed          = 0
        upper_body_cmd_msg.control_mode    = (4*np.ones(7).astype(int)).tolist() + \
                                             (4*np.ones(7).astype(int)).tolist() + \
                                             (4*np.ones(12).astype(int)).tolist() + \
                                             (4*np.ones(2).astype(int)).tolist() + \
                                             (4*np.ones(2).astype(int)).tolist()


        upper_body_cmd_msg.control_mode[5] = 5
        upper_body_cmd_msg.control_mode[6] = 5
        upper_body_cmd_msg.control_mode[12] = 5
        upper_body_cmd_msg.control_mode[13] = 5
        upper_body_cmd_msg.jointPosVec     = robot_30dof_solution
        # upper_body_cmd_msg.jointSpeedVec   = np.zeros_like(robot_30dof_solution).tolist()
        # if self.pre_pose is None:
        #     upper_body_cmd_msg.jointSpeedVec    = np.zeros_like(robot_30dof_solution).tolist()
        #     upper_body_cmd_msg.jointCurrentVec  = np.zeros_like(robot_30dof_solution).tolist()
        # else:
        #     speed = robot_30dof_solution - self.pre_pose
        #     upper_body_cmd_msg.jointSpeedVec    = speed.tolist()
        #     upper_body_cmd_msg.jointCurrentVec  = np.zeros_like(robot_30dof_solution).tolist()
        # self.pre_pose = robot_30dof_solution

        upper_body_cmd_msg.jointCurrentVec = np.zeros_like(robot_30dof_solution).tolist()
        upper_body_cmd_msg.jointKp = np.zeros_like(robot_30dof_solution).tolist()
        upper_body_cmd_msg.jointKd = np.zeros_like(robot_30dof_solution).tolist()

        # upper_body_cmd_msg.control_mode[14:20] = (20*np.ones(6).astype(int)).tolist()
        # upper_body_cmd_msg.control_mode[20:26] = (20*np.ones(6).astype(int)).tolist()

        upper_body_cmd_msg.jointPosVec = robot_30dof_solution.tolist()
        # upper_body_cmd_msg.jointPosVec[14] = 40
        # upper_body_cmd_msg.jointPosVec[20] = 40

        upper_body_cmd_msg.jointKp = (np.ones(30) * 40).tolist()
        upper_body_cmd_msg.jointKd = (np.ones(30) * 100).tolist()
        if self.speed_plan_pre_position is None:
            speed = np.zeros_like(robot_30dof_solution).tolist()
            acc = np.zeros_like(robot_30dof_solution).tolist()
            upper_body_cmd_msg.jointSpeedVec = np.zeros_like(robot_30dof_solution).tolist()
            upper_body_cmd_msg.jointCurrentVec = np.zeros_like(robot_30dof_solution).tolist()
            upper_body_cmd_msg.jointTorqueVec = np.zeros_like(robot_30dof_solution).tolist()
        else:
            if np.sum(self.speed_plan_pre_speed) == 0:
                speed = (robot_30dof_solution - self.speed_plan_pre_position) / (self.interpolation_period / 1000)
                acc = np.zeros_like(robot_30dof_solution)
                upper_body_cmd_msg.jointSpeedVec = speed.tolist()
                upper_body_cmd_msg.jointCurrentVec = np.zeros_like(robot_30dof_solution).tolist()
                # # 只计算了左臂的转矩输出 并且传入的模型和数据都只有7个关节 所以传入的位置 速度 加速度都只有 7位 并进行了低通滤波
                upper_body_cmd_msg.jointTorqueVec[:7] = np.zeros_like(robot_30dof_solution).tolist()
                # upper_body_cmd_msg.jointTorqueVec[:7] = np.array(
                #     upper_body_cmd_msg.jointTorqueVec[:7]) * alpha + (1 - alpha) * pinocchio.rnea(
                #     self.left_arm_pin_model, self.left_arm_pin_data, robot_30dof_solution[:7], speed[:7], acc[:7])

            else:
                speed = (robot_30dof_solution - self.speed_plan_pre_position) / (self.interpolation_period / 1000)
                acc = (speed - self.speed_plan_pre_speed) / (self.interpolation_period / 1000)
                upper_body_cmd_msg.jointSpeedVec = speed.tolist()
                upper_body_cmd_msg.jointCurrentVec = np.zeros_like(robot_30dof_solution).tolist()
                upper_body_cmd_msg.jointTorqueVec[:7] = np.zeros_like(robot_30dof_solution).tolist()

                # # 只计算了左臂的转矩输出 并且传入的模型和数据都只有7个关节 所以传入的位置 速度 加速度都只有 7位 并进行了低通滤波
                # upper_body_cmd_msg.jointTorqueVec[:7] = np.array(
                #     upper_body_cmd_msg.jointTorqueVec[:7]) * alpha + (1 - alpha) * pinocchio.rnea(
                #     self.left_arm_pin_model, self.left_arm_pin_data, robot_30dof_solution[:7], speed[:7], acc[:7])

        self.speed_plan_pre_position = np.copy(robot_30dof_solution)
        self.speed_plan_pre_speed = np.copy(speed)
        self.speed_plan_pre_acc = np.copy(acc)
        upper_body_cmd_msg.jointSpeedVec = np.array(upper_body_cmd_msg.jointSpeedVec)
        
        upper_body_cmd_msg.jointSpeedVec[5] = 0
        upper_body_cmd_msg.jointSpeedVec[6] = 0
        upper_body_cmd_msg.jointSpeedVec[12] = 0
        upper_body_cmd_msg.jointSpeedVec[13] = 0

        if self.gripper_selection[0]:
            if self.init_gripper_cnt < 19:
                upper_body_cmd_msg.control_mode[14:20] = (201 * np.ones(6).astype(int)).tolist()
                upper_body_cmd_msg.jointPosVec[14] = 500
            elif self.init_gripper_cnt < 39 and self.init_gripper_cnt >= 19:
                upper_body_cmd_msg.control_mode[14:20] = (30 * np.ones(6).astype(int)).tolist()
                upper_body_cmd_msg.jointPosVec[14] = 500
            elif self.init_gripper_cnt < 59 and self.init_gripper_cnt >= 39:
                upper_body_cmd_msg.control_mode[14:20] = (201 * np.ones(6).astype(int)).tolist()
                upper_body_cmd_msg.jointPosVec[14] = 500
            else:
                upper_body_cmd_msg.control_mode[14:20] = (20 * np.ones(6).astype(int)).tolist()
                upper_body_cmd_msg.jointSpeedVec[14] = 10
        else :
            if self.init_gripper_cnt < 9:
                upper_body_cmd_msg.control_mode[14:20] = (0 * np.ones(6).astype(int)).tolist()
            else:
                upper_body_cmd_msg.control_mode[14:20] = (4 * np.ones(6).astype(int)).tolist()
                
                
        if self.gripper_selection[1]:
            if self.init_gripper_cnt < 19:
                upper_body_cmd_msg.control_mode[20:26] = (201 * np.ones(6).astype(int)).tolist()
                upper_body_cmd_msg.jointPosVec[20] = 10
            elif self.init_gripper_cnt < 39 and self.init_gripper_cnt >= 19:
                upper_body_cmd_msg.control_mode[20:26] = (30 * np.ones(6).astype(int)).tolist()
                upper_body_cmd_msg.jointPosVec[20] = 1
            elif self.init_gripper_cnt < 59 and self.init_gripper_cnt >= 39:
                upper_body_cmd_msg.control_mode[20:26] = (201 * np.ones(6).astype(int)).tolist()
                upper_body_cmd_msg.jointPosVec[20] = 10
            else:
                upper_body_cmd_msg.control_mode[20:26] = (20 * np.ones(6).astype(int)).tolist()
                # upper_body_cmd_msg.jointSpeedVec[14] = 200
                upper_body_cmd_msg.jointSpeedVec[20] = 1
        else :
            if self.init_gripper_cnt < 9:
                upper_body_cmd_msg.control_mode[20:26] = (0 * np.ones(6).astype(int)).tolist()
            else:
                upper_body_cmd_msg.control_mode[20:26] = (4 * np.ones(6).astype(int)).tolist()

        self.init_gripper_cnt = self.init_gripper_cnt + 1
        if self.init_gripper_cnt > 100:
            self.init_gripper_cnt = 100
        return upper_body_cmd_msg

if __name__ =="__main__":
     s = lcmUnit()
