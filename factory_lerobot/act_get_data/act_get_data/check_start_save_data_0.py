import copy
import sys
import rclpy

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

        self.publisher_start_save_data = self.create_publisher(Int32, '/start_save_data', 10)


    def start_save_data(self, data):
        self.get_logger().info('Publishing : start save data\n' )
        data = Int32(data=data)
        self.publisher_start_save_data.publish(data)
    

def main():
    rclpy.init()
    Check_handler = robotControlRos()
    Check_handler.start_save_data(0)
    # 运行节点
    # rclpy.spin(Check_handler)
    #
    # # 销毁节点，退出ROS2
    Check_handler.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
