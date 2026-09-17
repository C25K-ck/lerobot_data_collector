import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
 
class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/image_data',
            self.image_callback,
            10)
        self.cv_bridge = CvBridge()
 
    def image_callback(self, msg):
        print("enter image callback")
        # 转换ROS图像到OpenCV图像
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # 显示图像
        cv2.imshow('ROS Image', cv_image)
        cv2.waitKey(1)
 
def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    rclpy.shutdown()
 
if __name__ == '__main__':
    main()