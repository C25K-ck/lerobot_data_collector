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
    0.08, 0.1, 1.0, -0.3, -1.0, 0.0, 0.1,   
    0.08, -0.1, -1.0, 0.3, 1.0, 0.0, 0.1,
    3.08, 3.08, 3.08, 3.08, 0.93, 2.86, 
    3.08, 3.07, 3.08, 3.07, 0.93, 2.87, 
    0.01, 0.0, -0.0, 0.0]

GO_HOME_READY = [
    0.08, 0.1, 1.0, -0.3, -1.0, 0.3, 0.1, 
    0.08, -0.1, -1.0, 0.3, 1.0, -0.3, 0.1,
    3.07, 3.08, 3.08, 3.08, 0.92, 2.72, 
    3.08, 3.07, 3.08, 3.07, 0.93, 2.74, 
    0.01, 0.0, 
    -0.0, -0.0]

def main():
    p2p_motion = P2PMotion()
    while(True):
        res = p2p_motion.get_current_pos()
        print(f"res: {res}")
        if res:
            '''
            回位
            '''
            # act_p2p(p2p_motion, p2p_motion.q, GO_HOME_READY)
            # print(f"go home")
            
            
            '''
            把手收回
            '''
            tt_pre = copy.copy(p2p_motion.q)
            res = p2p_motion.get_current_pos()
            tt = copy.copy(p2p_motion.q)
            tt[0] = 1
            tt[7] = 1
            tt[1] = 0.33
            tt[8] = -0.33
            tt[3] = -1.7
            tt[10] = 1.7
            
            
            # tt[0] = 0
            # tt[7] = 0
            # tt[3] = 0
            # tt[10] = 0
            
            formatted_numbers = [round(num, 2) for num in tt]
            print(f"p2p_motion.q: {formatted_numbers}")
            act_p2p(p2p_motion, tt_pre, tt)
            
            '''
            把肘子放下去
            '''
            tt_pre = copy.copy(tt)
            res = p2p_motion.get_current_pos()
            tt = copy.copy(p2p_motion.q)
            tt[0] = 0
            tt[7] = 0
            tt[3] = 0
            tt[10] = 0
            formatted_numbers = [round(num, 2) for num in tt]
            print(f"p2p_motion.q: {formatted_numbers}")
            act_p2p(p2p_motion, tt_pre, tt)
            '''
            把关节收回去
            '''
            tt_pre = copy.copy(tt)
            res = p2p_motion.get_current_pos()
            tt = copy.copy(p2p_motion.q)
            tt[0] = 0.03
            tt[7] = 0.03
            tt[2] = 0
            tt[4] = 0
            tt[9] = 0
            tt[11] = 0
            
            formatted_numbers = [round(num, 2) for num in tt]
            print(f"p2p_motion.q: {formatted_numbers}")
            act_p2p(p2p_motion, tt_pre, tt)
            '''
            到R
            '''
            tt_pre = copy.copy(tt)
            res = p2p_motion.get_current_pos()
            RECOVERPOS[14] = 0
            RECOVERPOS[20] = 0
            RECOVERPOS[5] = 0.52
            RECOVERPOS[12] = -0.52
            tt_pre[14] = 0 
            tt_pre[20] = 0 
            
            formatted_numbers = [round(num, 2) for num in RECOVERPOS]
            print(f"p2p_motion.q: {formatted_numbers}")
            act_p2p(p2p_motion, tt_pre, RECOVERPOS)
            
            
        
            break
        time.sleep(1)

if __name__ == "__main__":
    main()