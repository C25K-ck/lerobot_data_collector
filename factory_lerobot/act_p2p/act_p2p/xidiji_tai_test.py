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

HAND_OPEN = [
    3.07, 3.08, 3.08, 3.08, 0.92, 2.72, 
    3.08, 3.07, 3.08, 3.07, 0.93, 2.74, 
]

zhua_home_pos = [
    -0.04, 0.04, 1.26, -1.2, -1.27, 0.44, 0.1, 
    -0.16, 0.06, -1.35, 1.05, 1.42, -0.34, 0.08, 
    0.52, 1, 2.31, 2.5, 0.91, 1.57, 
    0.5, 2.3, 2.51, 2.5, 0.93, 1.57, 
    0.01, 0.0, -0.0, 0.0]

DAOWEI_POS = [
    -0.5, 0.04, 1.26, -1.4, -1.27, 0.44, 0.1, 
    -0.5, -0.07, -1.35, 1.4, 1.42, -0.34, 0.08, 
    0.52, 1, 2.31, 2.5, 0.91, 1.57, 
    0.5, 2.3, 2.51, 2.5, 0.93, 1.57, 
    0.01, 0.0, -0.0, 0.0]

PUTDOWN = [
    -0.3, 0.04, 1.26, -1.3, -1.27, 0.44, 0.1, 
    -0.3, -0.06, -1.35, 1.3, 1.42, -0.34, 0.08, 
    0.52, 1, 2.31, 2.5, 0.91, 1.57, 
    0.5, 2.3, 2.51, 2.5, 0.93, 1.57, 
    0.01, 0.0, -0.0, 0.0]

def main():
    p2p_motion = P2PMotion()
    while(True):
        res = p2p_motion.get_current_pos()
        if res:
            '''
            回位
            '''
            # tt = copy.copy(p2p_motion.q)
            hand_id_list = list(range(14,26))
            count = 0
            # for i in range(len(zhua_home_pos)):
            #     if i in hand_id_list:
            #         zhua_home_pos[i] = HAND_OPEN[count]
            #         count+=1
            
            act_p2p(p2p_motion, p2p_motion.q, DAOWEI_POS)
            
            # act_p2p(p2p_motion, p2p_motion.q, zhua_home_pos)
            # print(f"go home")
            
            # act_p2p(p2p_motion, zhua_home_pos, DAOWEI_POS)
            
            
            # act_p2p(p2p_motion, DAOWEI_POS, PUTDOWN)
            
            
            
            '''
            夹扫地机
            '''
            break
        time.sleep(1)

if __name__ == "__main__":
    main()