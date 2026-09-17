import argparse
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ManualSaveControl(Node):
    def __init__(self) -> None:
        super().__init__('manual_save_control')
        self.pub_main = self.create_publisher(String, '/start_save_data', 10)
        self.pub_orbbec = self.create_publisher(String, '/orbbec_camera_control', 10)
        self.pub_rs_left = self.create_publisher(String, '/realsense_left_control', 10)
        self.pub_rs_right = self.create_publisher(String, '/realsense_right_control', 10)
        self.pub_rs_head = self.create_publisher(String, '/realsense_head_control', 10)

    def publish_start(self, session_id: str, start_ts: float, frames: int) -> None:
        main_msg = String()
        main_msg.data = f"start:{session_id}:{start_ts:.6f}:{frames}"
        self.pub_main.publish(main_msg)

        cam_msg = String()
        cam_msg.data = f"start:camera_{session_id}:{start_ts:.6f}:{frames}"
        self.pub_orbbec.publish(cam_msg)
        self.pub_rs_left.publish(cam_msg)
        self.pub_rs_right.publish(cam_msg)

    def publish_stop(self) -> None:
        stop_msg = String()
        stop_msg.data = 'stop'
        self.pub_main.publish(stop_msg)
        self.pub_orbbec.publish(stop_msg)
        self.pub_rs_left.publish(stop_msg)
        self.pub_rs_right.publish(stop_msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Manual control for data collection start/stop (ROS2).')
    parser.add_argument('mode', choices=['start', 'stop', '0', '1'], help='start/0 = data==0, stop/1 = data==1')
    parser.add_argument('--delay', type=float, default=3.0, help='Delay seconds before start (for start mode)')
    parser.add_argument('--frames', type=int, default=2400, help='Target frames (for start mode)')
    parser.add_argument('--session', type=str, default=None, help='Session id (default: auto)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    is_start = args.mode in ('start', '0')
    session_id = args.session or f"manual_{int(time.time())}"

    rclpy.init()
    node = ManualSaveControl()

    try:
        if is_start:
            start_ts = time.time() + float(args.delay)
            node.get_logger().info(
                f"Start: session={session_id}, ts={start_ts:.6f}, frames={args.frames}")
            node.publish_start(session_id, start_ts, int(args.frames))
        else:
            node.get_logger().info('Stop')
            node.publish_stop()

        # 给ROS时间发送消息
        rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


