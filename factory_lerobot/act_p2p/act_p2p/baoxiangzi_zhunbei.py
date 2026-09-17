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

HAND_POS = [
    3.07, 3.08, 3.08, 3.08, 0.92, 2.83, 
    3.07, 3.08, 3.08, 3.08, 0.92, 2.83, 
]


def main():
    p2p_motion = P2PMotion()
    while(True):
        res = p2p_motion.get_current_pos()
        if res:
            init_pos = [
                0.0, 0.5,  1.7,-1.2, -1.57, 0.0, -0.0, 
                0.0, -0.5, -1.7, 1.2, 1.57, 0.0, -0.0, 
                3.07, 3.08, 3.08, 3.08, 0.92, 2.81, 
                3.07, 3.08, 3.08, 3.08, 0.92, 2.83, 
                0.01, -0.01, 0.0, 0.0]
            tt = copy.copy(init_pos)
            tt[1] = 0.5
            tt[8] = -0.5
            data = tt
            act_p2p(p2p_motion, p2p_motion.q, data)


            bao = [
                0.07, 0.31, 1.39, -1.56, -1.33, 0.16, 0.03, 
                0.15, -0.07, -1.59, 1.66, 1.41, -0.12, 0.07, 
                3.07, 3.08, 3.08, 3.08, 0.92, 2.8, 
                3.06, 3.07, 3.07, 3.07, 0.91, 2.79, 
                -0.0, 0.0, -0.0, -0.0]
            # tt = copy.copy(data)
            # tt = tt.tolist()
            # tt[1] = 0.5
            # tt[8] = -0.5
            # data = tt[0:14] + HAND_POS + [0,0,0,0]                    
            act_p2p(p2p_motion, data, bao)

            bao_zhijia = [
                0.07, 0.25, 1.39, -1.56, -1.33, 0.16, 0.03, 
                0.15, -0.07, -1.59, 1.66, 1.41, -0.02, 0.07, 
                2.6, 2.6, 2.6, 2.6, 0.92, 2.8, 
                2.6, 2.6, 2.6, 2.6, 0.91, 2.79, 
                -0.0, 0.0, -0.0, -0.0]
            act_p2p(p2p_motion, bao, bao_zhijia)

            bao_tai = [
                -0.13, 0.25, 1.39, -1.56, -1.33, 0.16, 0.03, 
                -0.15, -0.07, -1.59, 1.66, 1.41, -0.02, 0.07, 
                2.6, 2.6, 2.6, 2.6, 0.92, 2.8, 
                2.6, 2.6, 2.6, 2.6, 0.91, 2.79, 
                -0.0, 0.0, -0.0, -0.0]
            act_p2p(p2p_motion, bao_zhijia, bao_tai)

            break
        time.sleep(1)

if __name__ == "__main__":
    main()