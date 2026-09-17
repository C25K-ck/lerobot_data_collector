from __init__ import *
import rclpy
from act_mocap import *
import time
import psutil
import os

time.sleep(3)


def main():
    rclpy.init()
    robot_ctrl_ros = mocapActionInfo()

        # 运行节点
    rclpy.spin(robot_ctrl_ros)
    #
    # # 销毁节点，退出ROS2
    robot_ctrl_ros.destroy_node()
    rclpy.shutdown()


if __name__ =="__main__":
    current_process = psutil.Process(os.getpid())
    target_cores = [8,9,10,11]
    current_process.cpu_affinity(target_cores)
    main()
