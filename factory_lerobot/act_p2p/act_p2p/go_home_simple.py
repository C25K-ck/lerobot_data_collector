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
    0.08, 0.3, 1.2, -0.5, -1.5, 0.0, 0.1,   
    0.08, -0.3, -1.2, 0.5, 1.5, 0.0, 0.1,
    3.08, 3.08, 3.08, 3.08, 0.93, 2.86, 
    3.08, 3.07, 3.08, 3.07, 0.93, 2.87, 
    0.01, 0.0, -0.0, 0.0]

RECOVERPOS[26] = 0
RECOVERPOS[27] = 0

# INIT_QPOS =[0.10000000149011612, 0.20000000298023224, 1.0199999809265137, -0.3100000023841858, -1.0299999713897705, 0.30000001192092896, 0.10000000149011612, 0.354631676207114, -0.23269600693470616, -1.2581684716120114, 1.8045330562464819, 2.9145729153655298, -0.17757333860194244, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3400000035762787]

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
            tt_pre = copy.copy(p2p_motion.q)
            tt_pre[14] = RECOVERPOS[14]
            tt_pre[20] = RECOVERPOS[20]
            # tt_pre[20] = 200
            
            
            
            formatted_numbers = [round(num, 2) for num in RECOVERPOS]
            print(f"p2p_motion.q: {formatted_numbers}")
            act_p2p(p2p_motion, tt_pre, RECOVERPOS)
            
            break
        time.sleep(1)

if __name__ == "__main__":
    main()
