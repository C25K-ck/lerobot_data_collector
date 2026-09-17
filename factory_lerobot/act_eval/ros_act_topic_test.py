import os
import copy
import rclpy
from ros_act_eval import nodeActEval





def main():
    rclpy.init()
    act_eval_ros = nodeActEval()

        # 运行节点
    rclpy.spin(act_eval_ros)
    #
    # # 销毁节点，退出ROS2
    act_eval_ros.destroy_node()
    rclpy.shutdown()


if __name__ =="__main__":

    main()
