import rclpy
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




qos_profile = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,  # 或者 QoSReliabilityPolicy.RELIABLE
    history=QoSHistoryPolicy.KEEP_LAST,  # 或者 QoSHistoryPolicy.KEEP_ALL
    depth=10  # 根据需要调整队列大小
)



class nodeActEval(Node):
    def __init__(self):
        super().__init__('camera convert')


        self.remote_img_init()


    def remote_img_init(self):
        self.cv_bridge_head = CvBridge()
   
        
        self.subscription = self.create_subscription(SyncImg, "/multi_camera/sync_img",  self.image_head_callback, qos_profile=qos_profile)

        self.bridge_head_send = CvBridge()
        self.publisher_head_ = self.create_publisher(Image, '/rgbd3/color/image_raw', 10)
        print("init")


    def image_head_callback(self, msg):

        stream_imagef = np.frombuffer(msg.imgf_array, dtype=np.uint8)

        imagef  = cv2.imdecode(stream_imagef,  cv2.IMREAD_COLOR)
        size = (640, 480)
        imagef = cv2.resize(imagef, size)

        f_rgb = cv2.cvtColor(imagef, cv2.COLOR_BGR2RGB)



        data_head = self.bridge_head_send.cv2_to_imgmsg(f_rgb,encoding="bgr8") 

        print("pub cam")
        self.publisher_head_.publish(data_head)
def main():
    rclpy.init()
    cam_convert_ros = nodeActEval()

        # 运行节点
    rclpy.spin(cam_convert_ros)
    #
    # # 销毁节点，退出ROS2
    cam_convert_ros.destroy_node()
    rclpy.shutdown()


if __name__ =="__main__":

    main()
