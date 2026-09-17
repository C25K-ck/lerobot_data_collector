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
            tt = copy.copy(p2p_motion.q)
            tt = tt.tolist()
            data = tt[0:14] + HAND_POS + [0,0,0,0]                    
            # act_p2p(p2p_motion, p2p_motion.q, data)
            
            all_zero = [
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                3.07, 3.08, 3.08, 3.08, 0.92, 2.72, 
                3.08, 3.07, 3.08, 3.07, 0.93, 2.74, 
                0.01, 0.0, 
                -0.0, -0.0]
            act_p2p(p2p_motion, p2p_motion.q, all_zero)
            
            break
        time.sleep(1)

if __name__ == "__main__":
    main()