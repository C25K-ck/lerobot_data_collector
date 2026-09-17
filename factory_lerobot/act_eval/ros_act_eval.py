import copy
import sys

import numpy as np
import cv2

from rclpy.node import Node
from std_msgs.msg import String

from std_msgs.msg import Float32MultiArray, Int32
from cv_bridge import CvBridge

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CompressedImage

from multi_camera_msgs.msg import SyncImg
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from utils import get_image

qos_profile = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,  # 或者 QoSReliabilityPolicy.RELIABLE
    history=QoSHistoryPolicy.KEEP_LAST,  # 或者 QoSHistoryPolicy.KEEP_ALL
    depth=10  # 根据需要调整队列大小
)

class dataFlag():
    def __init__(self) -> None:
        self.camera_flag = False

        self.camera_waist_flag = False
        self.camera_detect_flag = False

class dataInfo():
    def __init__(self) -> None:
        # self.image_list = [np.zeros((640,480,3)), np.zeros((640,480,3)), np.zeros((640,480,3))]
        self.image_list = {
            # "headfl": np.zeros((640,480,3)),
            "headf":  np.zeros((480,640,3)),
            # "headfr": np.zeros((640,480,3)),
            # "left_hand": np.zeros((640,480,3)),
            "right_hand": np.zeros((480,640,3)),
            # "handr_depth": np.zeros((640,480,1))
        }

class nodeActEval(Node):
    def __init__(self):
        super().__init__('ACT_EVAL')

        self.publisher_qpos = self.create_publisher(Float32MultiArray, '/act_qpos', 10)
        
        self.publisher_task_mode = self.create_publisher(Int32, '/task_mode', 10)
        self.timer_period_get_data = self.create_timer(0.002, self.eval_bc)
        # self.timer_period_get_data = self.create_timer(0.5, self.eval_bc)

        self.publisher_control_glue_mashine = self.create_publisher(Int32, '/control_glue_mashine', 10)

        self.data_flag = dataFlag()
        self.data_info = dataInfo()

        self.remote_img_init()


    def remote_img_init(self):
        self.cv_bridge_head = CvBridge()
        self.cv_bridge_left_hand = CvBridge()
        self.cv_bridge_right_hand = CvBridge()        
        
        self.subscription = self.create_subscription(SyncImg, "/multi_camera/sync_img",  self.image_head_callback, qos_profile=qos_profile)
        # self.sub_left_hand_rgb = self.create_subscription(Image,'/rgbd1/color/image_rect_raw',self.image_left_hand_callback,10)
        self.sub_right_hand_rgb = self.create_subscription(Image,'/rgbd2/color/image_rect_raw',self.image_right_callback,10)

        # self.subscription_handr_depth = self.create_subscription(Image, '/rgbd2/depth/image_rect_raw', self.handr_depth_callback, 10)

        self.cv_bridge_depth = CvBridge()
        print("init")


    def image_head_callback(self, msg):
        # print("********************************************")
        # stream_imagefl = np.frombuffer(msg.imgfl_array, dtype=np.uint8)
        stream_imagef = np.frombuffer(msg.imgf_array, dtype=np.uint8)
        # stream_imagefr = np.frombuffer(msg.imgfr_array, dtype=np.uint8)
        # imagefl = cv2.imdecode(stream_imagefl, cv2.IMREAD_COLOR)
        imagef  = cv2.imdecode(stream_imagef,  cv2.IMREAD_COLOR)
        # imagefr = cv2.imdecode(stream_imagefr, cv2.IMREAD_COLOR)
        size = (640, 480)
        # imagefl = cv2.resize(imagefl, size)
        imagef = cv2.resize(imagef, size)
        # imagefr = cv2.resize(imagefr, size)
        # fl_rgb = cv2.cvtColor(imagefl, cv2.COLOR_BGR2RGB)
        f_rgb = cv2.cvtColor(imagef, cv2.COLOR_BGR2RGB)
        # fr_rgb = cv2.cvtColor(imagefr, cv2.COLOR_BGR2RGB)
        # self.data_info.image_list["headfl"] = fl_rgb
        self.data_info.image_list["headf"] = f_rgb
        # self.data_info.image_list["headfr"] = fr_rgb


        self.curr_image = get_image(self.data_info.image_list, ['headf', "right_hand"])

        
        self.data_flag.camera_flag = True
        self.data_flag.camera_detect_flag = True


        


    def image_left_hand_callback(self, msg):
        cv_image = self.cv_bridge_left_hand.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        size = [640, 480]
        cv_image = cv2.resize(cv_image, size)
        left_hand_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # self.data_info.image_list[1] = left_hand_rgb
        self.data_info.image_list["left_hand"] = left_hand_rgb
        # cv2.imshow("left hand", cv_image) 
        # cv2.waitKey(1)


    def image_right_callback(self, msg):
        cv_image = self.cv_bridge_right_hand.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        size = [640, 480]
        cv_image = cv2.resize(cv_image, size)
        right_hand_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        right_hand_rgb = cv_image
        right_rgb_flip = cv2.flip(right_hand_rgb, -1)

        self.data_info.image_list["right_hand"] = right_rgb_flip

        # right_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # right_rgb_flip = cv2.flip(right_rgb, -1)


        
        # self.data_info.image_list[2] = right_rgb_flip

        # cv2.imshow("right hand", cv_image) 
        # cv2.waitKey(10)

        self.data_flag.camera_waist_flag = True
        



    def handr_depth_callback(self, msg):

        
        pass


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
