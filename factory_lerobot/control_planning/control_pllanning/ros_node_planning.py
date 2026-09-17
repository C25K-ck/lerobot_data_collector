import copy
import sys

import numpy as np

from rclpy.node import Node

from std_msgs.msg import String

from std_msgs.msg import Float32MultiArray, Int32


class controlPlanRos(Node):
    def __init__(self):
        super().__init__('ROS_PLANNING')

        self.publisher_qpos = self.create_publisher(Float32MultiArray, '/act_qpos', 10)
        self.timer_period_get_data = self.create_timer(0.01, self.period_get_data)
        self.timer_period_iter_kinemic = self.create_timer(0.01, self.period_iter_kinemic)

        self.publisher_start_save_data = self.create_publisher(Int32, '/start_save_data', 10)

        self.publisher_change_task_mode = self.create_publisher(Int32, '/task_mode', 10)

        self.publisher_obj_pos_quat = self.create_publisher(Float32MultiArray, '/obj_pos_quat', 10)

    def pub_act_qpos(self, qpos):
        qpos_pub = Float32MultiArray(data=qpos)
        self.publisher_qpos.publish(qpos_pub)
        # self.get_logger().info('Publishing: "%s"' % qpos_pub)

        
    def period_get_data(self):
        pass

    def period_iter_kinemic(self):
        pass

    def timer_update_mocap_inf(self):
        pass

    def start_save_data(self, data):
        self.get_logger().info('Publishing : start save data\n' )
        data = Int32(data=data)
        self.publisher_start_save_data.publish(data)
    
    def publish_obj_pos_quat(self, pos_quat):
        pos_quat_pub = Float32MultiArray(data=pos_quat)
        self.publisher_obj_pos_quat.publish(pos_quat_pub)
        self.get_logger().info('Publishing: "%s"' % pos_quat_pub)

# if __name__ == '__main__':
#     main()
