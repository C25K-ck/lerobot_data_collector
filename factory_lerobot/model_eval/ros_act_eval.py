import copy
import sys

import numpy as np
import cv2

from rclpy.node import Node
from std_msgs.msg import String

from std_msgs.msg import Float32MultiArray, Int32
from cv_bridge import CvBridge

from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CompressedImage


class dataFlag():
    def __init__(self) -> None:
        self.camera_flag = False

class dataInfo():
    def __init__(self) -> None:
        self.image_list = [np.zeros((640,480,3)), np.zeros((640,480,3)), np.zeros((640,480,3))]


class nodeActEval(Node):
    def __init__(self):
        super().__init__('ACT_EVAL')

        self.publisher_qpos = self.create_publisher(Float32MultiArray, '/act_qpos', 10)
        
        self.timer_period_get_data = self.create_timer(0.02, self.eval_bc)

        self.data_flag = dataFlag()
        self.data_info = dataInfo()


    def remote_img_init(self):
        self.cv_bridge_head = CvBridge()
        self.cv_bridge_left_hand = CvBridge()
        self.cv_bridge_right_hand = CvBridge()

        self.sub_head_rgb = self.create_subscriptioself.data_flag.camera_flag = Falsen(Image,'/rgbd3/color/image_raw',self.image_head_callback,10)
        self.sub_left_hand_rgb = self.create_subscription(Image,'/rgbd1/color/image_rect_raw',self.image_left_hand_callback,10)
        self.sub_right_hand_rgb = self.create_subscription(Image,'/rgbd2/color/image_rect_raw',self.image_right_callback,10)
        print("init")

 
    def image_head_callback(self, msg):
        cv_image = self.cv_bridge_head.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        size = [640, 480]
        cv_image = cv2.resize(cv_image, size)
        head_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        self.data_info.image_list[0] = head_rgb
        cv2.imshow("head", cv_image) 
        cv2.waitKey(1)
        self.data_flag.camera_flag = True

    def image_left_hand_callback(self, msg):
        cv_image = self.cv_bridge_left_hand.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        size = [640, 480]
        cv_image = cv2.resize(cv_image, size)
        left_hand_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        self.data_info.image_list[1] = left_hand_rgb
        cv2.imshow("left hand", cv_image) 
        cv2.waitKey(1)


    def image_right_callback(self, msg):
        cv_image = self.cv_bridge_right_hand.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        size = [640, 480]
        cv_image = cv2.resize(cv_image, size)
        head_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        self.data_info.image_list[2] = head_rgb
        cv2.imshow("right hand", cv_image) 
        cv2.waitKey(1)


    def pub_act_qpos(self, qpos):
        qpos_pub = Float32MultiArray(data=qpos)
        self.publisher_qpos.publish(qpos_pub)

        
    def eval_bc(self):
        pass

    def timer_update_mocap_inf(self):
        pass

    def start_save_data(self, data):
        self.get_logger().info('Publishing : start save data\n' )
        data = Int32(data=data)
        self.publisher_start_save_data.publish(data)
    

# if __name__ == '__main__':
#     main()
