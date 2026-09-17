
import json
import copy
import math
import time
from kinemic_utils import *
import sys

from ros_node import *

import sys
import numpy as np
from p2p_motion import P2PMotion

# from keyboard_util import * 
from kinemic import loadUrdf
from mocap_data_manage import mocapDataManage
from ros_node import *
from teleop_pico import TeleopData
# from lcm_unit import lcmUnit
from lcm_unit_gripper import lcmUnit
from conf import *

from mocap_unit import mocapUnit, btnCtrlUnit




IS_REAL= True

class mocapManage():
    def __init__(self) -> None:

        self.wrist_l_m =  mocapUnit([[-0.9948, 0.9948], [-0.4538, 0.4538]], 2, 0.8)
        self.wrist_r_m =  mocapUnit([[-0.9948, 0.9948], [-0.4538, 0.4538]], 2, 0.8)
        self.perl_m = mocapUnit([[-0.2618, 0.2618], [0,0.5], [-1.5708, 1.5708]], 3, 0.01)
        self.head_m = mocapUnit([[-0.5236, 0.5236], [-0.3491, 0.1745]], 2, 1)

        self.gripper_l_m = btnCtrlUnit(0.5, [0, 80], 0)
        self.gripper_r_m = btnCtrlUnit(0.5, [0, 80], 0)

        # self.finger_l_m = [btnCtrlUnit(0.02, [0, 3], 3),btnCtrlUnit(0.02, [0, 3], 3),btnCtrlUnit(0.02, [0, 3], 3),btnCtrlUnit(0.02, [0, 3], 3),btnCtrlUnit(0.02, [-0.2, 0.92], 0.92),btnCtrlUnit(0.02, [1.57, 2.87], 2.87)]
        # self.finger_r_m = [btnCtrlUnit(0.02, [0, 3], 3),btnCtrlUnit(0.02, [0, 3], 3),btnCtrlUnit(0.02, [0, 3], 3),btnCtrlUnit(0.02, [0, 3], 3),btnCtrlUnit(0.02, [-0.2, 0.92], 0.92),btnCtrlUnit(0.02, [1.57, 2.87], 2.87)]

        self.finger_l_m = [btnCtrlUnit(0.02, [2.08, 3], 3),btnCtrlUnit(0.02, [2.08, 3], 3),btnCtrlUnit(0.02, [2.08, 3], 3),btnCtrlUnit(0.02, [2.08, 3], 3),btnCtrlUnit(0.02, [0.5, 0.92], 0.92),btnCtrlUnit(0.02, [1.77, 2.87], 2.87)]
        self.finger_r_m = [btnCtrlUnit(0.02, [2.08, 3], 3),btnCtrlUnit(0.02, [2.08, 3], 3),btnCtrlUnit(0.02, [2.08, 3], 3),btnCtrlUnit(0.02, [2.08, 3], 3),btnCtrlUnit(0.02, [0.5, 0.92], 0.92),btnCtrlUnit(0.02, [1.77, 2.87], 2.87)]


    def update_wrist_state(self,  wrist_l_p_e, wrist_r_p_e):
        wrist_l_quat = eul_to_quat(wrist_l_p_e[3:])
        wrist_r_quat = eul_to_quat(wrist_r_p_e[3:])

        self.wrist_l_m.move([wrist_l_quat[0], wrist_l_quat[2]])
        self.wrist_r_m.move([wrist_r_quat[0], -1 * wrist_r_quat[2]])
        


    def update_head_state(self, head_p_e):
        head_quat = eul_to_quat(head_p_e[3:])
        self.head_m.move([head_quat[2], -1*head_quat[0]])
        

    def update_perl_state(self, perl_p_e):
        perl_quat = eul_to_quat(perl_p_e[3:])

        # self.perl_m.move([-2*perl_quat[2],  -1 * perl_quat[0], 2*perl_quat[1]])
        self.perl_m.move([0,  -1 * perl_quat[0], 2*perl_quat[1]])

    def update_gripper_state(self,btn_l, btn_r):

        btn_idx = 2
        move_speed = 5
        if(btn_l[btn_idx]==True):
            self.gripper_l_m.move(move_speed)    


        elif(btn_l[btn_idx]==False):
            self.gripper_l_m.move(-move_speed)

        if(btn_r[btn_idx]==True):
            self.gripper_r_m.move(move_speed)

        elif(btn_r[btn_idx]==False):
            self.gripper_r_m.move(-move_speed)

    def update_finger_state(self,btn_l, btn_r):
        btn_idx_f = 2
        btn_idx_t = 3
        move_speed = 1


        if(btn_l[btn_idx_f]==True):
            for i in range(5):
                self.finger_l_m[i].move(-move_speed)

        elif(btn_l[btn_idx_f]==False):
            for i in range(5):
                self.finger_l_m[i].move(move_speed)

        if(btn_r[btn_idx_f]==True):
            for i in range(5):
                self.finger_r_m[i].move(-move_speed)

        elif(btn_r[btn_idx_f]==False):
            for i in range(5):
                self.finger_r_m[i].move(move_speed)


        if(btn_l[btn_idx_t]==True):
            self.finger_l_m[5].move(-1)

        elif(btn_l[btn_idx_t]==False):
            self.finger_l_m[5].move(1)

        if(btn_r[btn_idx_t]==True):
            self.finger_r_m[5].move(-1)

        elif(btn_r[btn_idx_t]==False):
            self.finger_r_m[5].move(1)
            

    def update_state(self, mocap_data):

        self.update_wrist_state(mocap_data.left_wrist_ctrl.deta_pos_eul, mocap_data.right_wrist_ctrl.deta_pos_eul)

        self.update_head_state(mocap_data.hmd_ctrl.deta_pos_eul)

        self.update_perl_state(mocap_data.chest_ctrl.deta_pos_eul)

        self.update_gripper_state(mocap_data.left_wrist_ctrl.btn, mocap_data.right_wrist_ctrl.btn)

        # self.update_finger_state(mocap_data.left_wrist_ctrl.btn, mocap_data.right_wrist_ctrl.btn)


    def get_wrist_state(self):
        return self.wrist_l_m.val[1], self.wrist_l_m.val[0], -1*self.wrist_r_m.val[1],self.wrist_r_m.val[0]
    
    def get_perl_state(self):
        return [self.perl_m.val[0], self.perl_m.val[1], self.perl_m.val[2]]  #ROLL PITCH YAW
    
    def get_head_state(self):
        return self.head_m.val
    
    def get_gripper_state(self):
        return self.gripper_l_m.val, self.gripper_r_m.val

    def get_finger_state(self):
        output_list = []
        for i in range(6):
            output_list.append(self.finger_l_m[i].val)
        for i in range(6):
            output_list.append(self.finger_r_m[i].val)

        return output_list

class mocapActionInfo(robotControlRos):
    def __init__(self):
        robotControlRos.__init__(self)

        self.pre_head = [0, 0]
        self.pre_perl = [0, 0, 0]
        self.pre_arm = [0 for _ in range(14)]
        self.pre_arm_vel = [0 for _ in range(14)]

        self.pre_wrist = [0,0,0,0]

        self.mocap_data_manage = mocapDataManage()

        self.kinemic_manage_l = loadUrdf(True)
        self.kinemic_manage_r = loadUrdf(False)

        self.teleop = TeleopData(TeleopData_IP)
        self.current_time = time.time()
        self.current_time_v2 = time.time()


        self.mocap_m = mocapManage()

        if(IS_REAL):
            self.lcm_unit = lcmUnit()
            self.p2p_m = P2PMotion(self.lcm_unit)

        self.is_start = False
        self.is_send_data = False


        self.wrist_pos = np.zeros(2*6)




    def period_get_data(self):

        ratio = 0.6

        if not self.is_start:
            return
        
        next(self.kinemic_manage_l.iteration)
        next(self.kinemic_manage_r.iteration)

        self.pub_robot_info(self.kinemic_manage_l.pos.tolist() + self.kinemic_manage_l.eul.tolist())        

        
        arm_list = [0 for _ in range(14)]
        qpos_mark_l = [1, 1, 1, 1, 1, 1, 1]
        qpos_mark_r = [1, 1, 1, 1, 1, 1, 1]
        for i in range(7):
            arm_list[i] = self.kinemic_manage_l.qpos[i] * qpos_mark_l[i]
            arm_list[i + 7] = self.kinemic_manage_r.qpos[i] * qpos_mark_r[i]

        # if(False):
        #     arm_list[4:7] = self.kinemic_manage_l.wrist_control()
        #     arm_list[11:14] = self.kinemic_manage_r.wrist_control()

        #     # self.kinemic_manage_l.wrist_control()
        #     # self.kinemic_manage_r.wrist_control()
        # # print(arm_list)

        arm_list, pre_arm_vel = self.interpolate_v3(self.pre_arm, arm_list, self.pre_arm_vel, ratio)
        self.pre_arm = copy.deepcopy(arm_list)
        self.pre_arm_vel = copy.deepcopy(pre_arm_vel)

        self.mocap_m.update_state(self.mocap_data_manage)

        wrist_list = self.mocap_m.get_wrist_state()

        wrist_list = self.interpolate_v2(self.pre_wrist, wrist_list, 0.3)
        
        self.pre_wrist = copy.deepcopy(wrist_list)

        arm_list[5:7],arm_list[12:14]  = wrist_list[:2], wrist_list[2:]

        mocap_data = [0 for _ in range(30)]
        arm_list_send = arm_list
        mocap_data[:14] = arm_list_send

        # mocap_data[26:28] = perl_val

        head_val = self.mocap_m.get_head_state()
        # mocap_data[28:30] = head_val

        perl_val = self.mocap_m.get_perl_state()
        perl_val[1] *= 3

        # mocap_data[26:28] = [perl_val[2], perl_val[0]]
        
        mocap_data[28:30] = [0, 0.34]

        # print(f"perl_val : {perl_val}\n")

        grriper_val = self.mocap_m.get_gripper_state()
        mocap_data[14] = grriper_val[0]
        mocap_data[20] = grriper_val[1]

        # finger_val = self.mocap_m.get_finger_state()
        # mocap_data[14:26] = finger_val

        # print(mocap_data[5:7])
        self.pub_act_qpos(mocap_data)


        # if self.is_send_data:
        # print(f"is send {self.is_send_data}\n")

        # 真机调试记得打开这里
        if(IS_REAL):
            if self.is_send_data:
            # if False:
                pass
                mocap_data = self.p2p_m.update_qpos(mocap_data)
                self.lcm_unit.send_to_robot(np.array(mocap_data))
                self.lcm_unit.wbc_floatbase_ctrl(perl_val)


    def update_mocap_inf(self):
        self.update_keyboard_state()

        mocap_info = self.teleop.update_mocap_info()
        self.mocap_data_manage.update_button(mocap_info)
        if(not self.is_start):
            return

        self.mocap_data_manage.parse_data(mocap_info)
        self.kinemic_manage_l.update_target_pos_diff(self.mocap_data_manage.left_arm_ctrl.deta_pos_eul)
        self.kinemic_manage_r.update_target_pos_diff(self.mocap_data_manage.right_arm_ctrl.deta_pos_eul)


        self.pub_mocap_info(self.mocap_data_manage.left_arm_ctrl.deta_pos_eul.tolist())


    def update_keyboard_state(self):
        self.current_time = time.time()

        if(self.current_time - self.current_time_v2<2):
            return


        if(self.mocap_data_manage.left_wrist_ctrl.btn[0]):
            if(IS_REAL):
                self.p2p_m.start_p2p_move()
            self.is_start = not self.is_start
            if(not self.is_start):
                self.is_send_data = False
                print("exit()")
                sys.exit()
            self.current_time_v2 =  self.current_time
            print(f"act qpos_ start : {self.is_start}!\n")
        if(self.is_start):
            if(self.mocap_data_manage.left_wrist_ctrl.btn[1]):
                self.is_send_data = not self.is_send_data
                self.current_time_v2 =  self.current_time

        if(self.mocap_data_manage.right_wrist_ctrl.btn[0]):
            self.start_save_data(0)
            self.current_time_v2 =  self.current_time
        if(self.mocap_data_manage.right_wrist_ctrl.btn[1]):
            self.start_save_data(1)
            self.current_time_v2 =  self.current_time



    @staticmethod
    def interpolate(pre_qpos, target_qpos, step):
        step_qpos = []
        for i in range(step):
            tmp_list = []
            for j in range(len(pre_qpos)):
                tmp_list.append((target_qpos[j] - pre_qpos[j]) / step * (i + 1) + pre_qpos[j])
            step_qpos.append(copy.deepcopy(tmp_list))
        return step_qpos

    @staticmethod
    def interpolate_v2(pre_qpos, target_qpos, ratio):
        step_qpos = [0 for _ in range(len(pre_qpos))]
        for i in range(len(pre_qpos)):
            step_qpos[i] = target_qpos[i] * ratio + pre_qpos[i] * (1 - ratio)
        return step_qpos

    @staticmethod
    def interpolate_v3(pre_qpos, target_qpos, pre_vel, ratio):
        vel = [0 for _ in range(len(target_qpos))]
        for i in range(len(target_qpos)):
            vel[i] = target_qpos[i] - pre_qpos[i]

        vel[i] = vel[i] * ratio + pre_vel[i] * (1 - ratio)

        step_qpos = [0 for _ in range(len(pre_qpos))]
        for i in range(len(target_qpos)):
            step_qpos[i] = pre_qpos[i] + vel[i]
        return step_qpos, vel
        vel = [0 for _ in range(len(target_qpos))]
        for i in range(len(target_qpos)):
            vel[i] = target_qpos[i] - pre_qpos[i]

        vel[i] = vel[i] * ratio + pre_vel[i] * (1 - ratio)

        step_qpos = [0 for _ in range(len(pre_qpos))]
        for i in range(len(target_qpos)):
            step_qpos[i] = pre_qpos[i] + vel[i]
        return step_qpos, vel
