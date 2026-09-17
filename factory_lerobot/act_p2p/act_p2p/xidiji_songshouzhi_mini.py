# !/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import time
from robot.seven_planner_humanoid import get_pos_list_seven_segment
from lcm_unit import lcmUnit
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
    print(f"d_len: {d_len}")
    handler.reset_pose()


RECOVERPOS =  [
    -0.04, 0.12, -0.0, -0.20, 0.03, 0.0, 0.0, 
    -0.04, -0.12, -0.0, 0.20, -0.03, 0.0, 0.0, 
    3.08, 3.08, 3.08, 3.08, 0.93, 2.86, 
    3.08, 3.07, 3.08, 3.07, 0.93, 2.87, 
    0.01, 0.0, -0.0, 0.0]

GO_HOME_READY = [
    0.03, 0.12, 1.55, -1.5, -1.55, 0.0, -0.0, 
    0.03, -0.13, -1.55, 1.5, 1.55, 0.0, -0.0, 
    3.07, 3.08, 3.08, 3.08, 0.92, 2.72, 
    3.08, 3.07, 3.08, 3.07, 0.93, 2.74, 
    0.01, 0.0, 
    -0.0, -0.0]

def main():
    p2p_motion = P2PMotion()
    while(True):
        res = p2p_motion.get_current_pos()
        if res:
            '''
            回位
            '''
            # act_p2p(p2p_motion, p2p_motion.q, GO_HOME_READY)
            print(f"go home")
            
            '''
            到准备抓的位置
            '''
            zhunbeizhua_pos = [
                0.34, 0.11, 1.57, -1.28, -1.57, 0.0, 0.0, 
                0.34, -0.12, -1.57, 1.28, 1.57, -0.0, -0.0, 
                3.08, 3.08, 3.08, 3.08, 0.92, 2.77, 
                3.08, 3.07, 3.08, 3.06, 0.93, 2.78, 
                0.01, 0.0, -0.0, -0.0]
            # act_p2p(p2p_motion, GO_HOME_READY, zhunbeizhua_pos)
            
            zhunbei_jia =  [
                0.1, 0.11, 1.26, -1.06, -1.27, 0.06, -0.13, 
                0.11, 0.0, -1.34, 1.14, 1.42, -0.21, 0.03, 
                3.08, 3.08, 3.08, 3.08, 0.92, 2.77, 
                3.08, 3.08, 3.08, 3.06, 0.93, 
                2.78, 0.01, 0.0, 0.0, -0.0]
            # act_p2p(p2p_motion, zhunbeizhua_pos, zhunbei_jia)
            
            zhunbei_jia_shouzhi = [
                0.1, 0.11, 1.26, -1.06, -1.27, 0.06, -0.13, 
                0.11, 0.0, -1.34, 1.14, 1.42, -0.21, 0.03, 
                3.08, 3.08, 3.08, 3.08, 0.92, 2.77, 
                3.08, 3.08, 3.08, 3.06, 0.93, 
                2.78, 0.01, 0.0, 0.0, -0.0]
            # act_p2p(p2p_motion, zhunbei_jia, zhunbei_jia_shouzhi)
            p2p_motion.get_current_pos()
            # act_p2p(p2p_motion, p2p_motion.q, zhunbei_jia_shouzhi)
            jia = [
                0.1, 0.03, 1.26, -1.05, -1.27, 0.25, -0.07, 
                0.11, 0.08, -1.35, 1.05, 1.42, -0.3, 0.05, 
                0.5, 1, 3.08, 3.08, 0.92, 2.77, 
                0.5, 1, 3.08, 3.06, 0.93, 2.78, 
                0.01, 0.0, 0.0, -0.0]
            # act_p2p(p2p_motion, zhunbei_jia_shouzhi, jia)
            print("zhunbei_jia_shouzhi")
            jia_muzhi = [
                0.1, 0.03, 1.26, -1.05, -1.27, 0.25, -0.07, 
                0.11, 0.08, -1.35, 1.05, 1.42, -0.3, 0.05, 
                0.5, 1, 3.08, 2.5, 2.5, 1.00, 
                0.5, 1, 3.08, 2.5, 2.5, 1.00, 
                0.01, 0.0, 0.0, -0.0]
            
            # act_p2p(p2p_motion, jia, jia_muzhi)
            print("jia_muzhi")
            
            
            act_p2p(p2p_motion, jia_muzhi, jia)
            act_p2p(p2p_motion, jia, zhunbei_jia_shouzhi)
            act_p2p(p2p_motion, zhunbei_jia_shouzhi, zhunbei_jia)
            act_p2p(p2p_motion, zhunbei_jia, zhunbeizhua_pos)
            
            
            '''
            夹扫地机
            '''
            break
        time.sleep(1)

if __name__ == "__main__":
    main()