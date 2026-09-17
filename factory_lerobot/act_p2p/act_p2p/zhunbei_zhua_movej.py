import time
import numpy as np
import math
import lcm
import threading
import os, sys
from os.path import dirname, join, abspath
from copy import deepcopy
import sys
import os

# 获取当前工作目录
pwd_path = os.getcwd()
# 往前后退 2 级
moca_path = os.path.dirname(os.path.dirname(pwd_path))
sys.path.append(moca_path)


from biped_lcm_types.python.upper_body_cmd_package import upper_body_cmd_package
from biped_lcm_types.python.upper_body_data_package import upper_body_data_package
from seven_segment_speed_plan import seven_segment_speed_plan


class RobotModel():
    def __init__(self) -> None:
        self.joint_position_dim = 30
        # upper_body_data_package包中的数据存储
        self.joint_current_pos_from_robot = None
        self.is_used_from_robot = None
        self.error_code_from_robot = None
        self.status_from_robot = None
        self.joint_current_speed_from_robot = None
        self.joint_current_current_or_torque_from_robot = None

        # 匹诺曹库初始化
        # urdf_path = join((dirname(str(abspath(__file__)))), "models/P5_left_arm.urdf")
        # mesh_dir = join((dirname(str(abspath(__file__)))), "meshes")
        # self.left_arm_pin_model, self.left_arm_pin_collision_model, self.left_arm_pin_visual_model = pinocchio.buildModelsFromUrdf(
        #     urdf_path, mesh_dir)
        # self.left_arm_pin_data = self.left_arm_pin_model.createData()
        # print('model name: ' + self.left_arm_pin_model.name)

        # LCM参数设置
        self.lcm_from_robot_period = 2  # 单位是ms
        self.lcm_form_robot_fs = 1000 / self.lcm_from_robot_period
        self.lcm = lcm.LCM('udpm://239.255.76.67:7667?ttl=1')
        self.lcm_thread_lock = threading.Lock()  # 创建一个锁
        self.lcm_thread_handle = threading.Thread(target=self.lcm_handle, daemon=True)
        self.lcm.subscribe('upper_body_data', self.upper_body_data_listener)
        self.lcm_thread_handle.start()

        # movej运动规划路径数据以及规划时的相关参数设置
        self.joint_delta_angle = None
        self.joint_delta_angle_max = None
        self.joint_delta_angle_index = None
        self.joint_movement_direction = None

        self.movej_plan_jerk_max = np.pi * 0.75
        self.movej_plan_acc_max = np.pi * 0.5
        self.movej_plan_speed_max = np.pi / 6
        self.movej_plan_target_position_list = None

        self.movej_plan_current_joint_position = None
        self.movej_plan_target_joint_position = None


        # movej/movel/movec插补需要的参数设置
        self.cart_interpolation_result = None
        self.cart_interpolation_position = np.zeros(3)
        self.cart_interpolation_pose = np.zeros((3, 3))
        self.interpolation_result = np.zeros(self.joint_position_dim)
        self.interpolation_period = 2  # 单位是ms
        self.inverse_kinematics_solution_success_flag = None
        # movej/movel中轨迹段数标志位
        # 因为第一段轨迹的第一个点需要是机器人获取的当前位置  但是第二段轨迹的第一个位置是第一段轨迹终止时的最后一个位置 而不能是当前时刻获取的机器人的位置 因为获取的位置不准
        # 其实第一段轨迹的第一个点获取的也不准 也会存在轻微的抖动 因为架构问题 暂时解决不了
        self.trajectory_segment_index = 0

        # LCM发送upper_body_cmd时需要的参数初始化
        self.speed_plan_pre_position = None
        self.speed_plan_pre_speed = None
        self.speed_plan_pre_acc = None

        # 数据处理最小值
        self.MIN_VAL = 0.0000001

        # csv文件点位下发周期
        self.csv_position_publish_period = 2  # 单位是ms

        # 上半身伺服模式设置 用于upper_body_cmd数据下发
        self.default_arm_control_mode = [4 for dim0 in range(14)]
        self.default_hand_control_mode = [4 for dim0 in range(12)]
        self.default_waist_control_mode = [4 for dim0 in range(2)]
        self.default_head_control_mode = [4 for dim0 in range(2)]

        # 两个手臂的六七关节伺服模式设置为模式5
        self.default_arm_control_mode[5] = 5
        self.default_arm_control_mode[6] = 5
        self.default_arm_control_mode[12] = 5
        self.default_arm_control_mode[13] = 5
        self.default_control_mode = self.default_arm_control_mode \
                                    + self.default_hand_control_mode \
                                    + self.default_waist_control_mode \
                                    + self.default_head_control_mode

    def lcm_handle(self):
        '''
        block func
        '''
        while True:
            self.lcm.handle()

    def upper_body_data_listener(self, channel, data):
        msg = upper_body_data_package.decode(data)
        with self.lcm_thread_lock:
            self.joint_current_pos_from_robot = np.array(msg.curJointPosVec)
            self.is_used_from_robot = np.array(msg.isUsed)
            self.error_code_from_robot = np.array(msg.curErrCodeVec)
            self.status_from_robot = np.array(msg.curStatusVec)
            self.joint_current_speed_from_robot = np.array(msg.curSpeedVec)
            self.joint_current_current_or_torque_from_robot = np.array(msg.curCurrentVec)

    def convert_to_arm_and_hand_cmd_package_msg(self, package):
        alpha = 0.0
        arm_and_hand_ctrl_msg = upper_body_cmd_package()
        arm_and_hand_ctrl_msg.isUsed = 0  # np.zeros_like(package).tolist()
        arm_and_hand_ctrl_msg.control_mode = self.default_control_mode
        arm_and_hand_ctrl_msg.jointPosVec = package.tolist()
        arm_and_hand_ctrl_msg.jointKp = (np.ones(30) * 40).tolist()
        arm_and_hand_ctrl_msg.jointKd = (np.ones(30) * 100).tolist()
        if self.speed_plan_pre_position is None:
            speed = np.zeros_like(package).tolist()
            acc = np.zeros_like(package).tolist()
            arm_and_hand_ctrl_msg.jointSpeedVec = np.zeros_like(package).tolist()
            arm_and_hand_ctrl_msg.jointCurrentVec = np.zeros_like(package).tolist()
            arm_and_hand_ctrl_msg.jointTorqueVec = np.zeros_like(package).tolist()


        else:
            if np.sum(self.speed_plan_pre_speed) == 0:
                speed = (package - self.speed_plan_pre_position) / (self.interpolation_period / 1000)
                acc = np.zeros_like(package)
                arm_and_hand_ctrl_msg.jointSpeedVec = speed.tolist()
                arm_and_hand_ctrl_msg.jointCurrentVec = np.zeros_like(package).tolist()
                # # 只计算了左臂的转矩输出 并且传入的模型和数据都只有7个关节 所以传入的位置 速度 加速度都只有 7位 并进行了低通滤波
                arm_and_hand_ctrl_msg.jointTorqueVec[:7] = np.zeros_like(package).tolist()
                # arm_and_hand_ctrl_msg.jointTorqueVec[:7] = np.array(
                #     arm_and_hand_ctrl_msg.jointTorqueVec[:7]) * alpha + (1 - alpha) * pinocchio.rnea(
                #     self.left_arm_pin_model, self.left_arm_pin_data, package[:7], speed[:7], acc[:7])

            else:
                speed = (package - self.speed_plan_pre_position) / (self.interpolation_period / 1000)
                acc = (speed - self.speed_plan_pre_speed) / (self.interpolation_period / 1000)
                arm_and_hand_ctrl_msg.jointSpeedVec = speed.tolist()
                arm_and_hand_ctrl_msg.jointCurrentVec = np.zeros_like(package).tolist()
                arm_and_hand_ctrl_msg.jointTorqueVec[:7] = np.zeros_like(package).tolist()

                # # 只计算了左臂的转矩输出 并且传入的模型和数据都只有7个关节 所以传入的位置 速度 加速度都只有 7位 并进行了低通滤波
                # arm_and_hand_ctrl_msg.jointTorqueVec[:7] = np.array(
                #     arm_and_hand_ctrl_msg.jointTorqueVec[:7]) * alpha + (1 - alpha) * pinocchio.rnea(
                #     self.left_arm_pin_model, self.left_arm_pin_data, package[:7], speed[:7], acc[:7])

        self.speed_plan_pre_position = np.copy(package)
        self.speed_plan_pre_speed = np.copy(speed)
        self.speed_plan_pre_acc = np.copy(acc)
        arm_and_hand_ctrl_msg.jointSpeedVec = np.array(arm_and_hand_ctrl_msg.jointSpeedVec)
        return arm_and_hand_ctrl_msg

    # --------------------------------------  movej speed plan and interpolation
    def cal_movej_plan_data(self, current_position, target_position):
        current_position = np.array(current_position)
        target_position = np.array(target_position)
        # 解决左臂单臂运行时当前位置第一个期望位置距离较近 但是运行时间较长的问题
        # 出现的原因是 通过LCM获取的当前位置 与期望的点位的位置之间可能存在较大的位置偏差 但是单臂的位置是接近的
        # 实际跑出来的效果就是 单臂没动 但是时间用在跑那些不跑的关节位移上
        # target_position[7:] = current_position[7:]
        self.movej_plan_current_joint_position = current_position
        # print("self.movej_plan_current_joint_position  = {} ".format(self.movej_plan_current_joint_position ))
        self.movej_plan_target_joint_position = target_position
        self.joint_delta_angle = np.zeros(current_position.shape)
        self.joint_movement_direction = np.zeros(current_position.shape)
        for i in range(self.joint_position_dim):
            self.joint_delta_angle[i] = target_position[i] - current_position[i]
            if self.joint_delta_angle[i] > self.MIN_VAL:
                self.joint_movement_direction[i] = 1
            else:
                self.joint_movement_direction[i] = -1

            self.joint_delta_angle[i] = np.fabs(self.joint_delta_angle[i])

        self.joint_delta_angle_max = np.max(self.joint_delta_angle)
        self.joint_delta_angle_index = np.argmax(self.joint_delta_angle)

    def movej_speed_plan_interpolation(self):
        for interpolation_time in np.arange(0, self.speed_plan.time_length, self.interpolation_period / 1000):
            start_time = time.time()  # 记录循环开始的时间
            if 0 <= interpolation_time <= self.speed_plan.accacc_time:
                self.speed_plan.cal_accacc_segment_data(interpolation_time)
            elif self.speed_plan.accacc_time < interpolation_time <= self.speed_plan.uniacc_time + self.speed_plan.accacc_time:
                interpolation_time = interpolation_time - self.speed_plan.accacc_time
                self.speed_plan.cal_uniacc_segment_data(interpolation_time)
            elif self.speed_plan.uniacc_time + self.speed_plan.accacc_time < interpolation_time <= self.speed_plan.acceleration_segment_time:
                interpolation_time = interpolation_time - (self.speed_plan.uniacc_time + self.speed_plan.accacc_time)
                self.speed_plan.cal_decacc_segment_data(interpolation_time)
            elif self.speed_plan.acceleration_segment_time < interpolation_time <= self.speed_plan.acceleration_segment_time + self.speed_plan.unispeed_time:
                interpolation_time = interpolation_time - self.speed_plan.acceleration_segment_time
                self.speed_plan.cal_unispeed_segment_data(interpolation_time)
            elif self.speed_plan.acceleration_segment_time + self.speed_plan.unispeed_time < interpolation_time <= self.speed_plan.acceleration_segment_time + self.speed_plan.unispeed_time + self.speed_plan.accdec_time:
                interpolation_time = interpolation_time - (
                            self.speed_plan.acceleration_segment_time + self.speed_plan.unispeed_time)
                self.speed_plan.cal_accdec_segment_data(interpolation_time)
            elif self.speed_plan.acceleration_segment_time + self.speed_plan.unispeed_time + self.speed_plan.accdec_time < interpolation_time <= self.speed_plan.time_length - self.speed_plan.decdec_time:
                interpolation_time = interpolation_time - (
                            self.speed_plan.acceleration_segment_time + self.speed_plan.unispeed_time + self.speed_plan.accdec_time)
                self.speed_plan.cal_unidec_segment_data(interpolation_time)
            else:
                interpolation_time = interpolation_time - (self.speed_plan.time_length - self.speed_plan.decdec_time)
                self.speed_plan.cal_decdec_segment_data(interpolation_time)

            for i in range(self.joint_position_dim):
                self.interpolation_result[i] = self.movej_plan_current_joint_position[
                                                   i] + self.speed_plan.cur_disp_normalization_ratio * \
                                               self.joint_movement_direction[i] * self.joint_delta_angle[i]

            # print("self.movej_plan_current_joint_position  = {} ".format(self.movej_plan_current_joint_position ))
            # print("self.interpolation_result  = {} ".format(self.interpolation_result ))
            # print("self.movej_plan_target_joint_position  = {} ".format(self.movej_plan_target_joint_position ))

            # with open("interpolate_trajectory.csv", 'a', newline='', encoding='utf-8') as csvfile:
            #     writer = csv.writer(csvfile)
            #     writer.writerow(self.interpolation_result)
            cmd_msg = self.convert_to_arm_and_hand_cmd_package_msg(self.interpolation_result)

            # 用于保证下发周期是4ms
            elapsed_time = (time.time() - start_time)  # 已经过的时间，单位是秒
            delay = max(0, self.interpolation_period / 1000 - elapsed_time)  # 4毫秒减去已经过的时间
            time.sleep(delay)  # 延迟剩余的时间

            self.lcm.publish('upper_body_cmd', cmd_msg.encode())

            # # 打印实际循环耗时，用于调试
            # total_elapsed_time = (time.time() - start_time)
            # print("Actual loop time: {:.4f} ms".format(total_elapsed_time * 1000))

        print("运行结束，到达目标点位！！！")

    def robot_movej_to_target_position(self):
        for i in range(len(self.movej_plan_target_position_list)):
            with self.lcm_thread_lock:
                if (self.trajectory_segment_index == 0):
                    current_joint_position = self.joint_current_pos_from_robot.copy()
                else:
                    current_joint_position = self.interpolation_result

                target_joint_position = self.movej_plan_target_position_list[self.trajectory_segment_index]
                self.cal_movej_plan_data(current_joint_position, target_joint_position)
                self.speed_plan = seven_segment_speed_plan(self.movej_plan_jerk_max, self.movej_plan_acc_max,
                                                           self.movej_plan_speed_max, self.joint_delta_angle_max)
                self.movej_speed_plan_interpolation()
                self.trajectory_segment_index = self.trajectory_segment_index + 1

    # --------------------------------------  movej speed plan and interpolation


if __name__ == "__main__":
    robot = RobotModel()
    hand_home_pos = np.array([165, 176, 176, 176, 25.0, 165.0, 165, 176, 176, 176, 25.0, 165.0],dtype = np.float64)
    hand_home_pos = list(hand_home_pos / 180 * np.pi)
    
    robot.movej_plan_target_position_list = [                                            
                                            [0.2, 0.12, 1.5707963, -0.2, -1.5707963, 0.0, 0.0, 0.2, -0.12, -1.5707963, 0.2, 1.5707963, 0.0, 0.0, 3.08, 3.08, 3.08, 3.08, 0.93, 2.86, 3.08, 3.07, 3.08, 3.07, 0.93, 2.87, 0.01, 0.0, -0.0, 0.0],
                                            [1, 0.5, 1.5707963, -1.7, -1.5707963, 0.0, 0.0, 1, -0.5, -1.5707963, 1.7, 1.5707963, 0.0, 0.0, 3.08, 3.08, 3.08, 3.08, 0.93, 2.86, 3.08, 3.07, 3.08, 3.07, 0.93, 2.87, 0.01, 0.0, -0.0, 0.0],
                                            [0.0, 0.2, 1.7, -1.2, -1.57, 0.0, -0.0, 0.0, -0.2, -1.7, 1.2, 1.57, 0.0, -0.0, 3.07, 3.08, 3.08, 3.08, 0.92, 2.81, 3.07, 3.08, 3.08, 3.08, 0.92, 2.83, 0.01, -0.01, 0.0, 0.0],
                                        ]
    time.sleep(1)
    robot.robot_movej_to_target_position()