# !/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import time
from robot.seven_planner_humanoid import get_pos_list_seven_segment
from lcm_unit_gripper import lcmUnit
import numpy as np
import copy


class P2PMotion():
    def __init__(self):
        self.lcm_unit = lcmUnit()
        self.speed = 45/180*math.pi
        self.data_list_t = None
        self.d_len = 0    
        self.q = [0 for i in range(30)]
        
    def get_current_pos(self):
        if self.lcm_unit.update_once_arm is True:
            self.q = self.lcm_unit.current_robot_state.q
            return True
        else:
            return False
    
    def get_pos_list(self, start, end):
        self.data_list_t, self.d_len = get_pos_list_seven_segment(start, end, self.speed)
        return self.data_list_t, self.d_len
        
    def reset_pose(self):
        # self.pub_act_qpos(self.qpos_numpy)

        for i in range(self.d_len):
            self.lcm_unit.send_to_robot(np.array(self.data_list_t[i]))
            time.sleep(0.004)

        # num_count = 5
        # for k in range(num_count):
        #     print(num_count-k)
        #     time.sleep(1)

def act_p2p(handler, start, end):
    data_list_t, d_len = handler.get_pos_list(start, end)
    handler.reset_pose()


RECOVERPOS =  [
    -0.04, 0.12, -0.0, -0.20, 0.03, 0.0, 0.0, 
    -0.04, -0.12, -0.0, 0.20, -0.03, 0.0, 0.0, 
    3.08, 3.08, 3.08, 3.08, 0.93, 2.86, 
    3.08, 3.07, 3.08, 3.07, 0.93, 2.87, 
    0.01, 0.0, -0.0, 0.0]

def main():
    p2p_motion = P2PMotion()
    while(True):
        res = p2p_motion.get_current_pos()
        if res:
            '''
            回位
            '''
            # act_p2p(p2p_motion, p2p_motion.q, RECOVERPOS)
            print(f"go ready")
            
            res = p2p_motion.get_current_pos()
            tt = copy.copy(p2p_motion.q)
            # tt[1] = 0.3
            tt[8] = -0.3
            # tt[5] = 0.52
            tt[12] = -0.52
            
            formatted_numbers = [round(num, 2) for num in tt]
            print(f"p2p_motion.q: {formatted_numbers}")
            act_p2p(p2p_motion, p2p_motion.q, tt)
            
            
            
            '''
            转 3 5关节, 保证4关节超前
            '''
            tt_pre = copy.copy(tt)
            res = p2p_motion.get_current_pos()
            tt = copy.copy(tt_pre)
            # tt[0] = 0.2
            tt[7] = 0.2
            # tt[2] = 1.5707963
            # tt[4] = -1.5707963
            tt[9] = -1.5707963
            tt[11] = 1.5707963
            # tt[5] = 0
            # tt[6] = 0
            tt[12] = 0
            
            formatted_numbers = [round(num, 2) for num in tt]
            print(f"p2p_motion.q: {formatted_numbers}")
            act_p2p(p2p_motion, tt_pre, tt)
            
            '''
            后拉 抬手
            0关节往后
            4关节往前 实际显示向地面
            '''
            tt_pre = copy.copy(tt)
            res = p2p_motion.get_current_pos()
            tt = copy.copy(tt_pre)
            # tt[0] = 1
            tt[7] = 1
            # tt[1] = 0.5
            tt[8] = -0.5
            # tt[3] = -1.7
            tt[10] = 1.7
            tt[12] = 0
            tt[26]=0.0
            
            formatted_numbers = [round(num, 2) for num in tt]
            print(f"p2p_motion.q: {formatted_numbers}")
            act_p2p(p2p_motion, tt_pre, tt)
            
            '''
            手往前推 
            
            
            0关节往后
            4关节往前 实际显示向地面
            '''
            tt_pre = copy.copy(tt)
            res = p2p_motion.get_current_pos()
            tt = copy.copy(tt_pre)
            # tt[0] = 0
            tt[7] = 0
            # tt[3] = -1.5
            tt[10] = 1.5
            tt[12] = 0
            tt[29] = 0.34
            tt[26]=0
            # tt[5] = 0.52
            tt[12] = -0.52
            formatted_numbers = [round(num, 2) for num in tt]
            print(f"p2p_motion.q: {formatted_numbers}")
            act_p2p(p2p_motion, tt_pre, tt)
            
            '''
            抱箱子
            '''
            # init_pos = [
            #     0.0, 0.5,  1.7,-1.2, -1.57, 0.0, -0.0, 
            #     0.0, -0.5, -1.7, 1.2, 1.57, 0.0, -0.0, 
            #     3.07, 3.08, 3.08, 3.08, 0.92, 2.81, 
            #     3.07, 3.08, 3.08, 3.08, 0.92, 2.83, 
            #     0.01, -0.01, 0.0, 0.0]
            # act_p2p(p2p_motion, tt, init_pos)
            
            break
        time.sleep(1)

if __name__ == "__main__":
    main()
