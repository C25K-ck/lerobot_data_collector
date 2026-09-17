import random
import numpy as np
import rclpy
from kinemic_utils import *
from kinemic import*
from ros_node_planning import controlPlanRos
from obj_control import objControl
import time

class taskPlanning(controlPlanRos):
    def __init__(self) -> None:
        controlPlanRos.__init__(self)
        self.max_step = 1000
        self.left_pos_eul = []
        self.right_pos_eul = []

        self.left_trajectory = []
        self.right_trajectory = []
        self.kinemic_manage_l = loadUrdf(True)
        self.kinemic_manage_r = loadUrdf(False)

        self.obj_manage = objControl()
        
        self.reset_task()
        self.init_para()

        self.is_start = True

    def init_para(self):
        self.pre_arm = [0 for _ in range(14)]
        self.pre_arm_vel = [0 for _ in range(14)]


    def reset_task(self):
        self.obj_manage.get_obj_pos_eul()
        self.get_init_grip_pos_eul()

        
    def update_obj_pos(self):
        # self.obj_manage.move_test_1()
        self.publish_obj_pos_quat(self.obj_manage.obj_pos+self.obj_manage.obj_quat)
    

    def get_init_grip_pos_eul(self):
        self.left_wrist_pos_eul = self.kinemic_manage_l.init_tp
        self.right_wrist_pos_eul = self.kinemic_manage_r.init_tp

        self.left_grip_pos_eul = self.left_wrist_pos_eul
        self.right_grip_pos_eul = self.right_wrist_pos_eul



    
    def get_oritation_obj_hand(self):
        oritation_l = [0,0,0]
        oritation_r = [0,0,0]
        for i in range(3):
            oritation_l[i] = self.obj_manage.obj_pos[i] - self.left_grip_pos_eul[i]
            oritation_r[i] = self.obj_manage.obj_pos[i] - self.right_grip_pos_eul[i]
        
        oritation_l[2] -= 0.96
        oritation_r[2] -= 0.96




        eul_l = self.eul_trans(oritation_to_eul(oritation_l), True)
        eul_r = self.eul_trans(oritation_to_eul(oritation_r), False)
        return eul_l, eul_r
    
    def get_pos_obj_hand(self):
        pos_l = [0,0,0]
        pos_r = [0,0,0]
        for i in range(3):
            pos_l[i] = self.obj_manage.obj_pos[i] - self.left_grip_pos_eul[i]
            pos_r[i] = self.obj_manage.obj_pos[i] - self.right_grip_pos_eul[i]

        pos_l[2] -= 1.16
        pos_r[2] -= 0.96

        # print(f"pos_l : {pos_l}, pos_r : {pos_r}\n")
        # import sys
        # sys.exit()

        return pos_l, pos_r
    
    def period_get_data(self):

        self.update_obj_pos()

        eul_l, eul_r = self.get_oritation_obj_hand()
        pos_l, pos_r = self.get_pos_obj_hand()


        deta_pos_eul_l = [pos_l[0],pos_l[1],pos_l[2], eul_l[0], eul_l[1], eul_l[2]]
        deta_pos_eul_r = [pos_r[0],pos_r[1],pos_r[2], eul_r[0], eul_r[1], eul_r[2]] 

        print(f"deta_pos_eul_l : {deta_pos_eul_l[3:]}")

        self.kinemic_manage_l.update_target_pos_diff(deta_pos_eul_l)
        self.kinemic_manage_r.update_target_pos_diff(deta_pos_eul_r)
        pass

    def period_iter_kinemic(self):
        ratio = 0.6
        if not self.is_start:
            return
        
        next(self.kinemic_manage_l.iteration)
        next(self.kinemic_manage_r.iteration)


        arm_list = [0 for _ in range(14)]
        qpos_mark_l = [1, 1, 1, 1, 1, 1, 1]
        qpos_mark_r = [1, 1, 1, 1, 1, 1, 1]
        for i in range(7):
            arm_list[i] = self.kinemic_manage_l.qpos[i] * qpos_mark_l[i]
            arm_list[i + 7] = self.kinemic_manage_r.qpos[i] * qpos_mark_r[i]

        arm_list, pre_arm_vel = self.interpolate_v3(self.pre_arm, arm_list, self.pre_arm_vel, ratio)
        self.pre_arm = copy.deepcopy(arm_list)
        self.pre_arm_vel = copy.deepcopy(pre_arm_vel)

        mocap_data = [0 for _ in range(30)]
        arm_list_send = arm_list
        mocap_data[:14] = arm_list_send

        self.pub_act_qpos(mocap_data)


    def eul_trans(self, eul, is_left=True):
        if is_left:
            rot_eul = eul_to_rot(np.array((0,0,0)))
            arm_rot_new = eul_to_rot(eul) @ rot_eul
            eul = rot_to_eul(arm_rot_new)
        else:
            rot_eul = eul_to_rot(np.array((0,0,0)))
            arm_rot_new = eul_to_rot(eul) @ rot_eul
            eul = rot_to_eul(arm_rot_new)
        return eul


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



def main():
    rclpy.init()
    planning_ctrl_ros = taskPlanning()

        # 运行节点
    rclpy.spin(planning_ctrl_ros)
    #
    # # 销毁节点，退出ROS2
    planning_ctrl_ros.destroy_node()
    rclpy.shutdown()


if __name__ =="__main__":

    main()