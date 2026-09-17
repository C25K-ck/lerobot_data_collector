
from __init__ import *

from act_get_data.act_get_data.init_cv import *
import rclpy
from act_get_data.act_get_data.save_manage_test import *
import time
import os
import psutil
time.sleep(3)

print("!")


def main():
    rclpy.init()
    save_data_m = saveDataMocap()

        # 运行节点
    rclpy.spin(save_data_m)
    #
    # # 销毁节点，退出ROS2
    save_data_m.destroy_node()
    rclpy.shutdown()


if __name__ =="__main__":
    current_process = psutil.Process(os.getpid())
    target_cores = [8,9,10,11]
    current_process.cpu_affinity(target_cores)
    main()
