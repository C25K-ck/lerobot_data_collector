import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, '/image_head_data', 10)
        self.timer = self.create_timer(0.1, self.publish_camera)
        self.cv_bridge = CvBridge()
 
        # 初始化相机
        self.cap = cv2.VideoCapture(0)  # 0 是默认相机的设备索引
        print("初始化相机")
 
    def publish_camera(self):
        ret, frame = self.cap.read()
        cv2.imshow("111",frame)
        cv2.waitKey(1)
        if not ret:
            self.get_logger().error('Failed to grab camera frame')
            return
 
        msg = self.cv_bridge.cv2_to_imgmsg(frame, 'bgr8')
        self.publisher_.publish(msg)
        print("pub")
 
def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CameraPublisher()
    rclpy.spin(camera_publisher)
    camera_publisher.cap.release()
    rclpy.shutdown()
 
if __name__ == '__main__':
    main()