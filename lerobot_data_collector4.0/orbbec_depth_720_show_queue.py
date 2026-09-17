import cv2

from pyorbbecsdk import *

import sys
import time

import numpy as np
import copy

import queue
import threading

import os
import psutil
import sys

# ROS2导入
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from pyorbbecsdk import *

# 导入frame_to_bgr_image函数
import sys
import os
# 添加robot_kinemic模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'robot_kinemic', 'robot_kinemic'))
from utils import frame_to_bgr_image

import threading

# 仅增加线程绑核（读取/写入线程）
def _set_current_thread_affinity(cores):
    try:
        if not cores:
            return
        tid = threading.get_native_id()
        os.sched_setaffinity(tid, set(cores))
    except Exception:
        pass

ORBBEC_READ_CORES = [4,5,6,7]
ORBBEC_WRITE_CORES = [13,14,15]
class AsyncImageDisplayer(Node):
    def __init__(self, target_fps=30):
        super().__init__('orbbec_camera')

        # 相机状态
        self.is_collecting = False
        self.collection_start_time = None
        self.should_stop = False
        self.current_session = None
        self.target_frames = 0
        self.start_timestamp = None
        
        # 订阅相机控制话题
        self.subscription = self.create_subscription(
            String,
            '/orbbec_camera_control',
            self.camera_control_callback,
            10
        )

        self.display_queue = queue.Queue(maxsize=1)  # 限制队列大小避免内存溢出

        # 大队列用于保存，采集不阻塞
        try:
            _qsz_env = os.getenv('ORBBEC_SAVE_QUEUE')
            _qsz = int(_qsz_env) if _qsz_env is not None else 4096
        except Exception:
            _qsz = 4096
        self.save_queue = queue.Queue(maxsize=_qsz)

        self.target_fps = target_fps
        self.min_interval = 1.0 / target_fps  # 每帧最小间隔（秒）
        self.last_display_time = 0.0
        self.window_name = "head Display"
        self.is_running = True
        self.last_timestamp = 0
        self.display_toggle = 0  # 15Hz显示：每2帧显示1帧
        
        # 启动显示线程
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        self.save_thread = threading.Thread(target=self._save_loop, daemon=True)
        self.save_thread.start()

    def _resize_for_display(self, img, ):
        
        return cv2.resize(img, (640, 360))

    def _display_loop(self):
        """显示线程主循环"""
        _set_current_thread_affinity(ORBBEC_READ_CORES)
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        while self.is_running:
            try:
                # 带超时的get，避免永久阻塞
                img, timestamp = self.display_queue.get(timeout=0.1)
                
                # 15Hz显示：每2帧显示1帧，降低显示负载
                self.display_toggle ^= 1
                if self.display_toggle == 0:
                    display_img = self._resize_for_display(img)
                    cv2.imshow(self.window_name, display_img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.stop()
                
                # 标记任务完成
                self.display_queue.task_done()
                
            except queue.Empty:
                # 队列空时继续循环
                continue

    def _save_loop(self):
        """保存线程：从单一队列取(color, depth, ts)并写盘"""
        _set_current_thread_affinity(ORBBEC_WRITE_CORES)
        last_session = None
        frame_count = 0
        timestamp_file = None
        while self.is_running:
            try:
                # 带超时的get，避免永久阻塞
                item = self.save_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                image_color, image_depth, rgb_timestamp = item

                # 检查是否需要开始新的采集会话
                if self.is_collecting and self.current_session != last_session:
                    # 创建新会话的输出目录 - 使用统一的images文件夹结构
                    last_session = self.current_session
                    
                    # 优先使用环境变量中的FIXED_REPO_ID（由主程序在开始采集时设置）
                    # 如果没有，则生成新的时间戳
                    import os
                    fixed_repo_id = os.getenv("FIXED_REPO_ID")
                    if fixed_repo_id:
                        repo_id = fixed_repo_id
                    else:
                        # 如果没有环境变量，则生成新的时间戳
                        from datetime import datetime
                        try:
                            from conf import TASK_CODE
                            task_code = TASK_CODE
                        except ImportError:
                            task_code = "data_collection"
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        repo_id = f"{task_code}_{timestamp}"
                    dataset_path = f"/home/dreame/data/hf_dataset/{repo_id}"
                    
                    # 查找data文件夹中的episode信息
                    data_folder = os.path.join(dataset_path, "data", "chunk-000")
                    if os.path.exists(data_folder):
                        # 查找data文件夹中的episode文件
                        existing_episodes = []
                        for item in os.listdir(data_folder):
                            if item.startswith("episode_") and item.endswith(".parquet"):
                                try:
                                    episode_num = int(item.split("_")[1].split(".")[0])
                                    existing_episodes.append(episode_num)
                                except:
                                    continue
                        
                        if existing_episodes:
                            latest_episode_num = max(existing_episodes) + 1  # 使用下一个编号
                            episode_folder = os.path.join(dataset_path, "chunk_000", f"episode_{latest_episode_num:06d}")
                        else:
                            episode_folder = os.path.join(dataset_path, "chunk_000", "episode_000000")
                    else:
                        episode_folder = os.path.join(dataset_path, "chunk_000", "episode_000000")
                    
                    # 创建images子文件夹结构
                    images_folder = os.path.join(episode_folder, "images")
                    rgb_dir = os.path.join(images_folder, "headf_rgbd_color")
                    depth_dir = os.path.join(images_folder, "headf_rgbd_depth")

                    

                    os.makedirs(rgb_dir, exist_ok=True)
                    os.makedirs(depth_dir, exist_ok=True)
                    
                    # 创建时间戳文件 - 放在对应的图像文件夹里
                    timestamp_file = os.path.join(rgb_dir, "headf_rgbd_timestamps.txt")
                    # 清空之前的时间戳文件
                    if os.path.exists(timestamp_file):
                        os.remove(timestamp_file)
                    
                    frame_count = 0
                    print(f"Orbbec使用episode目录: {episode_folder}")
                    print(f"Orbbec时间戳文件: {timestamp_file}")

                # 采集处理（如果正在采集且有frames）
                if self.is_collecting and not self.should_stop:
                    try:
                        current_time = rgb_timestamp/1000/1000/1000
                        
                        # 如果设置了开始时间戳，等待到指定时间
                        if self.start_timestamp and current_time < self.start_timestamp:
                            remaining_wait = self.start_timestamp - current_time
                            if frame_count == 0:  # 只在第一次显示等待信息
                                print(f"Orbbec等待{remaining_wait:.1f}秒后开始采集...")
                            if os.path.exists(timestamp_file):
                                os.remove(timestamp_file)
                            continue
                        
                        # 第一次开始采集时记录实际开始时间
                        if frame_count == 0:
                            self.collection_start_time = current_time
                            print(f"Orbbec开始采集，目标帧数: {self.target_frames}")
                        
                        # 检查是否达到目标帧数
                        if self.target_frames > 0 and frame_count >= self.target_frames:
                            print(f"Orbbec达到目标帧数{self.target_frames}，自动停止")
                            self.stop_collection()
                            continue
                        
                        # 备用超时保护（如果没有设置目标帧数）
                        if self.target_frames == 0 and self.collection_start_time:
                            duration = 60.0  # 默认60秒
                            if current_time - self.collection_start_time >= duration:
                                print(f"Orbbec采集时间到达{duration}秒，自动停止")
                                self.stop_collection()
                                continue

                        # 使用简单数字命名保存图像
                        timestamp_ns = rgb_timestamp

                        image_color_path = os.path.join(rgb_dir, f"{frame_count}.jpg")
                        image_depth_path = os.path.join(depth_dir, f"{frame_count}.png")

                        cv2.imwrite(image_color_path, image_color)
                        cv2.imwrite(image_depth_path, image_depth)
                        
                        # 保存时间戳到txt文件
                        if timestamp_file:
                            with open(timestamp_file, 'a') as f:
                                f.write(f"{timestamp_ns}\n")
                        
                        frame_count += 1
                        
                        # 定期报告进度
                        if frame_count % 30 == 0 and self.collection_start_time:
                            elapsed = rgb_timestamp - self.collection_start_time
                            fps = frame_count / elapsed if elapsed > 0 else 0
                            
                            if self.target_frames > 0:
                                progress = (frame_count / self.target_frames) * 100
                                remaining_frames = self.target_frames - frame_count
                                print(f"Orbbec: {frame_count}/{self.target_frames}帧 ({progress:.1f}%), {fps:.1f}fps, 剩余{remaining_frames}帧")
                        
                    except Exception as e:
                        print(f"orbbec save failed: {e}")
                else:
                    pass
            except Exception as e:
                print(f"orbbec save failed: {e}")
            finally:
                try:
                    self.save_queue.task_done()
                except Exception:
                    pass

    def show_image(self, img):
        """向队列中添加待显示的图像（主线程调用）"""
        if not self.is_running:
            return
        try:
            # 仅保留最新的图像（避免旧数据堆积）
            if self.display_queue.full():
                self.display_queue.get_nowait()  # 丢弃最旧的未处理图像
            self.display_queue.put((img, time.time()))
        except Exception as e:
            print(f"显示队列异常: {e}")


    def save_and_show(self, img, depth):
        if not self.is_running:
            return
        try:
            # 仅保留最新的图像（避免旧数据堆积）
            if self.display_queue.full():
                self.display_queue.get_nowait()  # 丢弃最旧的未处理图像

            timestamp = time.time_ns()
            self.display_queue.put((img, timestamp))
            # 保存队列非阻塞提交，满则丢弃最旧，保证采集不卡
            try:
                self.save_queue.put_nowait((img, depth, timestamp))
            except queue.Full:
                try:
                    _ = self.save_queue.get_nowait()
                    self.save_queue.task_done()
                except Exception:
                    pass
                try:
                    self.save_queue.put_nowait((img, depth, timestamp))
                except Exception:
                    pass


        except Exception as e:
            print(f"显示队列异常: {e}")

    def stop(self):
        """停止显示线程"""
        self.is_running = False
        self.display_thread.join(timeout=1.0)
        try:
            self.save_queue.join()
        except Exception:
            pass
        try:
            self.save_thread.join(timeout=1.0)
        except Exception:
            pass
        cv2.destroyAllWindows()


    def camera_control_callback(self, msg):
        """处理相机控制指令"""
        command = msg.data
        self.get_logger().info(f'Orbbec收到指令: {command}')
        
        if command.startswith("start:"):
            # 解析格式: "start:session_id:start_timestamp:target_frames"
            parts = command.split(":")
            if len(parts) >= 4:
                session_id = parts[1]
                start_timestamp = float(parts[2])
                target_frames = int(parts[3])
                self.start_collection(session_id, start_timestamp, target_frames)
            else:
                # 兼容旧格式
                session_id = command.split(":", 1)[1]
                self.start_collection(session_id)
        elif command == "stop":
            self.stop_collection()
    
    def start_collection(self, session_id, start_timestamp=None, target_frames=None):
        """开始采集"""
        # if self.is_collecting:
        #     self.get_logger().warning("Orbbec已在采集中")
        #     return
        
        self.current_session = session_id
        # 将start_timestamp转换为毫秒，与save_and_show中的时间戳单位一致
        self.start_timestamp = (start_timestamp or time.time())
        self.target_frames = target_frames or 0
        self.is_collecting = True
        self.should_stop = False
        
        if start_timestamp:
            delay = start_timestamp - time.time()
            self.get_logger().info(f"Orbbec准备采集，会话: {session_id}")
            self.get_logger().info(f"等待{delay:.1f}秒后开始，目标帧数: {target_frames}")
        else:
            self.get_logger().info(f"Orbbec开始采集，会话: {session_id}")
    
    def stop_collection(self):
        """停止采集"""
        if not self.is_collecting:
            self.get_logger().warning("Orbbec未在采集中")
            return
        
        self.is_collecting = False
        self.should_stop = True
        self.get_logger().info("Orbbec停止采集")





def _orbbec_camera_loop_head():
    """头部奥比中光相机采集循环"""
    rclpy.init()
    displayer = AsyncImageDisplayer(target_fps=30)
    pipeline = None
    
    # 重连尝试计数器
    reconnect_count = 0
    
    # 自定义分辨率设置
    COLOR_WIDTH = 1280
    COLOR_HEIGHT = 720
    COLOR_FPS = 30
    DEPTH_WIDTH = 1280
    DEPTH_HEIGHT = 720
    DEPTH_FPS = 30
    
    # cnt = 0
    save_count = 0
    time_use = time.time()
    cache_count = 0
    time_start = time_use
    
    # 30帧时间统计
    frame_count_30 = 0
    start_time_30 = None
    while True:
        try:
            # === 奥比中光设备初始化 ===
            pipeline = Pipeline()
            config = Config()
            
            # 获取彩色流配置 - 自定义分辨率
            profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            # 尝试获取指定分辨率的配置，如果失败则使用默认配置
            try:
                color_profile = profile_list.get_video_stream_profile(COLOR_WIDTH, COLOR_HEIGHT, OBFormat.RGB, COLOR_FPS)
                print(f"✓ 使用自定义彩色流配置: {COLOR_WIDTH}x{COLOR_HEIGHT} RGB {COLOR_FPS}fps")
            except:
                color_profile = profile_list.get_default_video_stream_profile()
                print("✓ 使用默认彩色流配置")
            config.enable_stream(color_profile)
            
            # 获取深度流配置 - 自定义分辨率
            profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            try:
                depth_profile = profile_list.get_video_stream_profile(DEPTH_WIDTH, DEPTH_HEIGHT, OBFormat.Y16, DEPTH_FPS)
                print(f"✓ 使用自定义深度流配置: {DEPTH_WIDTH}x{DEPTH_HEIGHT} Y16 {DEPTH_FPS}fps")
            except:
                depth_profile = profile_list.get_default_video_stream_profile()
                print("✓ 使用默认深度流配置")
            config.enable_stream(depth_profile)
            
            # 启用硬件D2C对齐
            config.set_align_mode(OBAlignMode.HW_MODE)
            print("启用硬件D2C对齐")
            
            # 启动pipeline
            pipeline.start(config)
            print("头部奥比中光相机连接成功")
            # 关闭RGB自动曝光并设置手动曝光（微秒）
            try:
                device = pipeline.get_device()
                sensors = device.get_sensor_list()
                for i in range(sensors.get_sensor_count()):
                    s = sensors.get_sensor(i)
                    if s.get_sensor_type() == OBSensorType.COLOR_SENSOR:
                        s.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
                        s.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, 20000)
                        break
            except Exception:
                pass
            
            
            # === 图像采集循环 ===
            while rclpy.ok():
                rclpy.spin_once(displayer, timeout_sec=0.001)
                try:
                    # 等待帧数据
                    frames = pipeline.wait_for_frames(1000)
                    if not frames:
                        continue
                    if cache_count<30:
                        cache_count+=1
                        time_use = time.time()
                        time_start = time_use
                        continue
                    
                    # 获取原始帧
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    
                    if not color_frame or not depth_frame:
                        continue
                    
                    # 使用硬件D2C对齐：直接使用获取的帧（已经在硬件层面对齐）
                    aligned_color_frame = color_frame
                    aligned_depth_frame = depth_frame
                    
                    
                    # 检查对齐后的帧
                    if not aligned_color_frame:
                        continue
                    
                    # 处理彩色图像
                    color_image = frame_to_bgr_image(aligned_color_frame)
                    if color_image is None:
                        continue
                    
                    # 图像翻转
                    # color_image = cv2.flip(color_image, -1)  # -1表示水平和垂直翻转
                    
                    # 处理深度数据
                    try:
                        depth_data = np.frombuffer(aligned_depth_frame.get_data(), dtype=np.uint16).reshape(
                            (aligned_depth_frame.get_height(), aligned_depth_frame.get_width()))
                    except ValueError:
                        continue
                    
                    # # 深度数据处理
                    depth_data = depth_data.astype(np.float32) * aligned_depth_frame.get_depth_scale()
                    depth_data = np.where((depth_data > 20) & (depth_data < 10000), depth_data, 0)
                    depth_data = depth_data.astype(np.uint16)


                    
                    # 深度图像翻转
                    # depth_data = cv2.flip(depth_data, -1)  # -1表示水平和垂直翻转
                    
                    # 赋值到原有数据结构
                    # self.data_info.image_list[5] = copy.copy(color_image)
                    # self.data_info.image_list[6] = copy.copy(depth_data)
                    
                    # 设置标志和时间戳
                    '''
                    self.data_flag.camera_flag = True
                    self.data_flag.head_camera_flag = True
                    '''
                    displayer.save_and_show(color_image, depth_data)

                    # 30帧时间统计
                    if start_time_30 is None:
                        start_time_30 = time.time()
                        frame_count_30 = 0
                        print("开始30帧时间统计...")
                    
                    frame_count_30 += 1
                    
                    if frame_count_30 >= 30:
                        end_time_30 = time.time()
                        elapsed_time = end_time_30 - start_time_30
                        fps = 30 / elapsed_time
                        print(f"获取30张图片耗时: {elapsed_time:.3f}秒, 平均FPS: {fps:.2f}")
                        
                        # 重置计数器，开始下一轮统计
                        start_time_30 = time.time()
                        frame_count_30 = 0

                    # cnt+=1
                    # save_count+=1
                    # # print(f"head 30fps time use: {time_use - time.time()}")
                    # if cnt>=30:
                    #     print(f"head 30fps time use: {time_use - time.time()}, save count: {save_count}")
                    #     time_use = time.time()
                    #     cnt = 0
                    # if save_count>2460:
                    #     print(f"exit, time use: {time_start-time.time()}")
                    #     sys.exit(0)
                    time.sleep(0.01)
                    
                except Exception as e:
                    # 设备中断时抛出异常（类似RealSense）
                    if "Frame didn't arrive" in str(e) or "timeout" in str(e).lower():
                        print(f"检测到设备连接中断: {e}")
                        break
                    else:
                        print(f"采集错误: {e}")
                        continue
                
        except (KeyboardInterrupt, SystemExit):
            # 正常退出
            print("程序退出...")
            break
            
        except Exception as e:
            # 设备错误处理
            reconnect_count += 1
            print(f"头部奥比中光相机错误: {e}")
            print(f"重连尝试 #{reconnect_count}...")
            
            # 清理资源
            if pipeline:
                try:
                    pipeline.stop()
                except:
                    pass
                pipeline = None
            
            # 延迟重连
            time.sleep(5)
            
        finally:
            # 清理资源
            if pipeline:
                try:
                    pipeline.stop()
                except:
                    pass



if __name__ == "__main__":
    current_process = psutil.Process(os.getpid())
    target_cores = [4,5,6,7]
    current_process.cpu_affinity(target_cores)
    _orbbec_camera_loop_head()
