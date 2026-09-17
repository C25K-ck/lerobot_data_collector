
import sys
import os

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
        self.default_arm_control_mode   = [5 for i in range(14)]
        self.default_hand_control_mode  = [4 for dim0 in range(12)]
        self.default_waist_control_mode = [5 for dim0 in range(2) ]
        self.default_head_control_mode = [5 for dim0 in range(2) ]
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


    def send_to_robot(self,data):
        upper_body_cmd_msg = self.load_upper_body_cmd_package(data)
        self.publish(self.upper_body_cmd_topic, upper_body_cmd_msg.encode())
        print(f"send data: {self.upper_body_cmd_topic, upper_body_cmd_msg}\n")

    def load_upper_body_cmd_package(self, robot_30dof_solution):
        upper_body_cmd_msg  = upper_body_cmd_package()
        upper_body_cmd_msg.isUsed          = 0
        upper_body_cmd_msg.control_mode    = (5*np.ones(7).astype(int)).tolist() + \
                                             (5*np.ones(7).astype(int)).tolist() + \
                                             (4*np.ones(12).astype(int)).tolist() + \
                                             (5*np.ones(2).astype(int)).tolist() + \
                                             (5*np.ones(2).astype(int)).tolist()

        upper_body_cmd_msg.jointPosVec     = robot_30dof_solution
        upper_body_cmd_msg.jointSpeedVec   = np.zeros_like(robot_30dof_solution).tolist()
        upper_body_cmd_msg.jointKp = np.zeros_like(robot_30dof_solution).tolist()
        upper_body_cmd_msg.jointKd = np.zeros_like(robot_30dof_solution).tolist()

        upper_body_cmd_msg.jointSpeedVec    = np.zeros_like(robot_30dof_solution).tolist()
        upper_body_cmd_msg.jointCurrentVec  = np.zeros_like(robot_30dof_solution).tolist()  

        upper_body_cmd_msg.jointCurrentVec = np.zeros_like(robot_30dof_solution).tolist()

        upper_body_cmd_msg.control_mode[14:20] = (20*np.ones(6).astype(int)).tolist()
        upper_body_cmd_msg.control_mode[20:26] = (20*np.ones(6).astype(int)).tolist()
        return upper_body_cmd_msg



if __name__ =="__main__":
     s = lcmUnit()
