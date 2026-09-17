import copy
import sys

import numpy as np

from sensor_msgs.msg import JointState
from geometry_msgs.msg  import PoseStamped, Point, Quaternion
from std_msgs.msg import Header
from rclpy.node import Node
from std_msgs.msg import String

from std_msgs.msg import Float32MultiArray, Int32


class robotControlRos(Node):
    def __init__(self):
        super().__init__('ROBOT_CONTROL')



        self.publisher_qpos = self.create_publisher(Float32MultiArray, '/act_qpos', 10)
        self.timer_period_get_data = self.create_timer(0.01, self.period_get_data)
        self.timer_update_mocap_inf = self.create_timer(0.02, self.update_mocap_inf)

        self.publisher_start_save_data = self.create_publisher(String, '/start_save_data', 10)

        self.publisher_change_task_mode = self.create_publisher(Int32, '/task_mode', 10)

        # 三个相机控制publisher
        self.publisher_orbbec_control = self.create_publisher(String, '/orbbec_camera_control', 10)
        self.publisher_realsense_left_control = self.create_publisher(String, '/realsense_left_control', 10)
        self.publisher_realsense_right_control = self.create_publisher(String, '/realsense_right_control', 10)

        self.publisher_mocap_info  = self.create_publisher(Float32MultiArray, '/mocap_info', 10)
        self.publisher_robot_info  = self.create_publisher(Float32MultiArray, '/robot_info', 10)

        self.publisher_joint_state = self.create_publisher(JointState, '/joint_states', 10)

        self.publisher_goal_pose = self.create_publisher(PoseStamped, '/goal_pose', 10)


    def pub_act_qpos(self, qpos):
        qpos_pub = Float32MultiArray(data=qpos)
        self.publisher_qpos.publish(qpos_pub)
        # self.get_logger().info('Publishing: "%s"' % qpos_pub)


    def pub_mocap_info(self, mocap_info):
        mocap_info_pub = Float32MultiArray(data=mocap_info)
        self.publisher_mocap_info.publish(mocap_info_pub)



    def pub_robot_info(self, robot_info):
        robot_info_pub = Float32MultiArray(data=robot_info)
        self.publisher_robot_info.publish(robot_info_pub)

        
    def period_get_data(self):
        pass

    def timer_update_mocap_inf(self):
        pass

    def start_save_data(self, data):
        self.get_logger().info('Publishing : start save data\n' )
        
        # 🎥 集成时间戳和帧数控制逻辑
        if data == 0:  # 开始采集
            import time
            session_id = f"data_sync_{int(time.time())}"
            delay_seconds = 3.0
            target_frames = 2400
            
            # 计算实际开始时间戳
            start_timestamp = time.time() + delay_seconds
            
            self.get_logger().info(f'启动数据采集: {session_id}')
            self.get_logger().info(f'延迟{delay_seconds}秒后开始，目标帧数: {target_frames}')
            
            # 发布数据采集指令（新格式：包含时间戳和帧数）
            data_msg = String()
            data_msg.data = f"start:{session_id}:{start_timestamp:.6f}:{target_frames}"
            self.publisher_start_save_data.publish(data_msg)
            
            # 同时发布给三个相机
            camera_msg = String()
            camera_msg.data = f"start:camera_{session_id}:{start_timestamp:.6f}:{target_frames}"
            
            self.publisher_orbbec_control.publish(camera_msg)
            self.publisher_realsense_left_control.publish(camera_msg)
            self.publisher_realsense_right_control.publish(camera_msg)
            
        elif data == 1:  # 停止采集
            self.get_logger().info('停止数据采集和三相机采集')
            
            # 发布停止数据采集指令
            data_msg = String()
            data_msg.data = "stop"
            self.publisher_start_save_data.publish(data_msg)
    
            # 发布停止相机采集指令
            camera_msg = String()
            camera_msg.data = "stop"
            
            self.publisher_orbbec_control.publish(camera_msg)
            self.publisher_realsense_left_control.publish(camera_msg)
            self.publisher_realsense_right_control.publish(camera_msg)

    def publish_joint_rviz_state(self, act_qpos, foot_qpos= [0. for _ in range(12)]):
        """
        Publish the joint states to ROS topic.
        """
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [
            'JOINT_HIP_ROLL_L', 'JOINT_HIP_YAW_L', 'JOINT_HIP_PITCH_L',
            'JOINT_KNEE_PITCH_L', 'JOINT_ANKLE_PITCH_L', 'JOINT_ANKLE_ROLL_L',
            'JOINT_HIP_ROLL_R', 'JOINT_HIP_YAW_R', 'JOINT_HIP_PITCH_R',
            'JOINT_KNEE_PITCH_R', 'JOINT_ANKLE_PITCH_R', 'JOINT_ANKLE_ROLL_R',
            'joint_wr', 'joint_wy', 'joint_hy', 'joint_hp',
            'joint_la1', 'joint_la2', 'joint_la3', 'joint_la4', 'joint_la5', 'joint_la6', 'joint_la7',
            'joint_ra1', 'joint_ra2', 'joint_ra3', 'joint_ra4', 'joint_ra5', 'joint_ra6', 'joint_ra7'
        ]

        # rot_trans = np.zeros((3,3))
        # rot_trans[0,0] = 1
        # rot_trans[1,2] = 1
        # rot_trans[2,1] = 1
        # per_eul = rot_to_eul((eul_to_rot(rot_trans @ pos_eul[3:]) )).tolist()

        # perl_pos = [0., 0., 0.]
        # per_eul = [0., 0., 0.]

        positions =  foot_qpos + act_qpos[26:30] + act_qpos[:14] 
        positions = list(map(float, positions))

        msg.position = positions
        msg.velocity = [0.0] * 30 # Default effort to 0

        msg.effort = [0.0] * 30 # Default effort to 0

        self.publisher_joint_state.publish(msg)
        # self.get_logger().debug(f'Published joint states: positions={positions}')


# if __name__ == '__main__':
#     main()
