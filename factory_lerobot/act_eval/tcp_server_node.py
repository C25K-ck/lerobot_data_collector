import socket
import threading
import rclpy
from rclpy.node import Node
import asyncio
from rclpy.node import Node
import time
from std_msgs.msg import String, Int32

class TCPServerNode(Node):
    def __init__(self):
        super().__init__('tcp_server_node')
        self.declare_parameter('host', '0.0.0.0')
        # self.declare_parameter('host', '192.168.12.75')
        self.declare_parameter('port', 15000)
        host = self.get_parameter('host').get_parameter_value().string_value
        port = self.get_parameter('port').get_parameter_value().integer_value
        
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((host, port))
        self.server.listen(5)
        self.get_logger().info(f"TCP Server started on {host}:{port}")

        self.is_send_data = False

        self.subscription_control_machine = self.create_subscription(Int32, '/control_glue_mashine', self.control_glue_machine_callback, 10)




    def control_glue_machine_callback(self, msg):
        print("get control command")
        self.is_send_data = True
        pass

# 继电器1闭合：11 05 00 00 FF 00 8E AA
# 继电器1断开：11 05 00 00 00 00 CF 5A
    async def handle_clients(self):


            try:
                self.client_socket, addr = self.server.accept()
                self.get_logger().info(f"New client connected: {addr}")
                while True:

                    await asyncio.sleep(0.5)

                    if(self.is_send_data):
                        self.client_socket.send(b'\x11\x05\x00\x00\xFF\x00\x8E\xAA')  # 返回响应
                        self.is_send_data = False
                        print("send data")
                        rec = self.client_socket.recv(4096)
                        print(rec)
                        time.sleep(1)
                        self.client_socket.send(b'\x11\x05\x00\x00\00\x00\xCF\x5A')  # 返回响应

                        rec = self.client_socket.recv(4096)
                        print(rec)
            except Exception as e:
                self.get_logger().error(f"Client handling error: {str(e)}")
            finally:
                self.client_socket.close()

    def __del__(self):
        self.client_socket.close()
        self.server.close()


async def ros_spin(node):

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.002)
            await asyncio.sleep(0.002)

    except Exception  as e:
        print(e)
    finally:
        executor.shutdown()
                
def main(args=None):
    
    rclpy.init(args=args)
    node = TCPServerNode()

    coro = asyncio.gather(ros_spin(node), node.handle_clients())
    # coro = asyncio.gather(node.handle_clients(), rclpy.spin(node))

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(coro)
    except KeyboardInterrupt:
        pass
    finally:

        rclpy.shutdown()

if __name__ == '__main__':
    main()