import cv2
from pyorbbecsdk import *
import sys
import time
import numpy as np
import queue
import threading
import os
import psutil
import ctypes
# ROS2导入
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# 导入frame_to_bgr_image函数
import sys
import os
# 添加robot_kinemic模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'robot_kinemic', 'robot_kinemic'))
from utils import frame_to_bgr_image


ORBBEC_READ_CORES_DEFAULT = [4,5,6,7]
ORBBEC_WRITE_CORES_DEFAULT = [13,14,15]

def _parse_cores_env(var_name: str, default_list):
    val = os.getenv(var_name)
    if not val:
        return default_list
    try:
        parts = [int(x) for x in val.replace(',', ' ').split() if x.strip().isdigit()]
        return parts if parts else default_list
    except Exception:
        return default_list

def _set_current_thread_affinity(cores):
    try:
        if not cores:
            return
        tid = threading.get_native_id()
        os.sched_setaffinity(tid, set(cores))
    except Exception:
        pass

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

        # 单一保存队列，避免双队列不同步
        _qsz_env = os.getenv('ORBBEC_SAVE_QUEUE')
        try:
            _qsz = int(_qsz_env) if _qsz_env is not None else 512
        except Exception:
            _qsz = 512
        self.save_queue = queue.Queue(maxsize=_qsz)

        self.target_fps = target_fps
        self.min_interval = 1.0 / target_fps  # 每帧最小间隔（秒）
        self.last_display_time = 0.0
        self.window_name = "head Display"
        self.is_running = True
        self.last_timestamp = 0
        # 线程亲和核可通过环境变量覆盖
        self.read_cores = _parse_cores_env("ORBBEC_READ_CORES", ORBBEC_READ_CORES_DEFAULT)
        self.write_cores = _parse_cores_env("ORBBEC_WRITE_CORES", ORBBEC_WRITE_CORES_DEFAULT)

        # 控制日志
        try:
            self.verbose = os.getenv('ORBBEC_VERBOSE', '0').lower() in ('1','true','yes')
        except Exception:
            self.verbose = False

        # 保持写盘质量为OpenCV默认值，避免改变原始图像质量

        # 预分配缓冲区（按需在保存线程里初始化/复用）
        self._bgr_buf = None  # np.ndarray[h, w, 3], uint8
        
        # 启动显示线程
        # 限制OpenCV内部线程，避免与我们自建线程竞争
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        self.save_thread = threading.Thread(target=self._save_loop, daemon=True)
        self.save_thread.start()

    def _resize_for_display(self, img):
        return cv2.resize(img, (640, 360))

    def _display_loop(self):
        """显示线程主循环"""
        _set_current_thread_affinity(self.read_cores)
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        while self.is_running:
            try:
                # 带超时的get，避免永久阻塞
                img, timestamp = self.display_queue.get(timeout=0.1)
                
                current_time = time.time()
                if current_time - self.last_display_time >= self.min_interval:
                    display_img = self._resize_for_display(img)
                    cv2.imshow(self.window_name, display_img)
                    self.last_display_time = current_time
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.stop()
                
                # 标记任务完成
                self.display_queue.task_done()
                
            except queue.Empty:
                # 队列空时继续循环
                continue

    def _save_loop(self):
        """保存线程主循环：队列项包含(color_src, depth_src, timestamp_ns)"""
        _set_current_thread_affinity(self.write_cores)
        last_session = None
        frame_count = 0
        timestamp_file = None
        timestamp_fh = None
        rgb_dir = None
        depth_dir = None
        collection_start_ns = None
        while self.is_running:
            try:
                # 带超时的get，避免永久阻塞
                item = self.save_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                color_src, depth_src, timestamp_ns = item

                # 结束请求时关闭时间戳句柄
                if self.should_stop and timestamp_fh is not None:
                    try:
                        timestamp_fh.close()
                    except Exception:
                        pass
                    timestamp_fh = None

                # 检查是否需要开始新的采集会话
                if self.is_collecting and self.current_session != last_session:
                    # 创建新会话的输出目录 - 使用统一的images文件夹结构
                    last_session = self.current_session
                    from conf import REPO_ID
                    dataset_path = f"/home/dreame/data/hf_dataset/{REPO_ID}"
                    data_folder = os.path.join(dataset_path, "data", "chunk-000")
                    if os.path.exists(data_folder):
                        existing_episodes = []
                        for item_name in os.listdir(data_folder):
                            if item_name.startswith("episode_") and item_name.endswith(".parquet"):
                                try:
                                    episode_num = int(item_name.split("_")[1].split(".")[0])
                                    existing_episodes.append(episode_num)
                                except Exception:
                                    continue
                        if existing_episodes:
                            latest_episode_num = max(existing_episodes) + 1
                            episode_folder = os.path.join(dataset_path, "chunk_000", f"episode_{latest_episode_num:06d}")
                        else:
                            episode_folder = os.path.join(dataset_path, "chunk_000", "episode_000000")
                    else:
                        episode_folder = os.path.join(dataset_path, "chunk_000", "episode_000000")
                    images_folder = os.path.join(episode_folder, "images")
                    rgb_dir = os.path.join(images_folder, "headf_rgbd_color")
                    depth_dir = os.path.join(images_folder, "headf_rgbd_depth")
                    os.makedirs(rgb_dir, exist_ok=True)
                    os.makedirs(depth_dir, exist_ok=True)
                    timestamp_file = os.path.join(rgb_dir, "headf_rgbd_timestamps.txt")
                    # 重新创建时间戳文件
                    try:
                        if os.path.exists(timestamp_file):
                            os.remove(timestamp_file)
                    except Exception:
                        pass
                    try:
                        timestamp_fh = open(timestamp_file, 'w')
                    except Exception:
                        timestamp_fh = None
                    frame_count = 0
                    collection_start_ns = None
                    if self.verbose:
                        print(f"Orbbec使用episode目录: {episode_folder}")
                        print(f"Orbbec时间戳文件: {timestamp_file}")

                # 采集处理（如果正在采集且有frames）
                if self.is_collecting and not self.should_stop:
                    # 起始时间（ns）
                    if frame_count == 0:
                        # 如果设置了开始时间戳（秒），等待到指定时间
                        if self.start_timestamp and (timestamp_ns / 1e9) < self.start_timestamp:
                            remaining_wait = self.start_timestamp - (timestamp_ns / 1e9)
                            if self.verbose:
                                print(f"Orbbec等待{remaining_wait:.1f}秒后开始采集...")
                            # 丢弃直到达到开始时间
                            continue
                        collection_start_ns = timestamp_ns
                        if self.verbose:
                            print(f"Orbbec开始采集，目标帧数: {self.target_frames}")

                    # 检查是否达到目标帧数
                    if self.target_frames > 0 and frame_count >= self.target_frames:
                        if self.verbose:
                            print(f"Orbbec达到目标帧数{self.target_frames}，自动停止")
                        self.stop_collection()
                        continue

                    # 转换彩色图像
                    try:
                        if hasattr(color_src, 'get_data'):
                            h, w = color_src.get_height(), color_src.get_width()
                            if self._bgr_buf is None or self._bgr_buf.shape != (h, w, 3):
                                self._bgr_buf = np.empty((h, w, 3), dtype=np.uint8)
                            rgb_view = np.frombuffer(color_src.get_data(), dtype=np.uint8).reshape((h, w, 3))
                            # 使用OpenCV SIMD路径做颜色转换到预分配缓冲
                            cv2.cvtColor(rgb_view, cv2.COLOR_RGB2BGR, dst=self._bgr_buf)
                            color_image = self._bgr_buf
                        else:
                            color_image = color_src
                    except Exception:
                        continue

                    # 转换深度图像到uint16（mm）, 并过滤范围
                    try:
                        if hasattr(depth_src, 'get_data'):
                            depth_data = np.frombuffer(depth_src.get_data(), dtype=np.uint16).reshape((depth_src.get_height(), depth_src.get_width()))
                            depth_data = depth_data.astype(np.float32) * depth_src.get_depth_scale()
                            depth_data = np.where((depth_data > 20) & (depth_data < 10000), depth_data, 0)
                            depth_data = depth_data.astype(np.uint16)
                        else:
                            depth_data = depth_src
                    except Exception:
                        continue

                    # 保存到磁盘
                    image_color_path = os.path.join(rgb_dir, f"{frame_count}.jpg")
                    image_depth_path = os.path.join(depth_dir, f"{frame_count}.png")
                    try:
                        cv2.imwrite(image_color_path, color_image)
                        cv2.imwrite(image_depth_path, depth_data)
                    except Exception as e:
                        print(f"orbbec save failed: {e}")

                    # 保存时间戳到txt文件
                    if timestamp_fh:
                        try:
                            timestamp_fh.write(f"{int(timestamp_ns)}\n")
                        except Exception:
                            pass

                    frame_count += 1

                    # 定期报告进度
                    if frame_count % 30 == 0 and collection_start_ns:
                        elapsed_s = max((timestamp_ns - collection_start_ns) / 1e9, 1e-6)
                        fps = frame_count / elapsed_s
                        if self.target_frames > 0:
                            progress = (frame_count / self.target_frames) * 100
                            remaining_frames = self.target_frames - frame_count
                            print(f"Orbbec: {frame_count}/{self.target_frames}帧 ({progress:.1f}%), {fps:.1f}fps, 剩余{remaining_frames}帧")
                
            except Exception as e:
                if self.verbose:
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


    def enqueue_save(self, color_src, depth_src, timestamp_ns):
        """非阻塞提交保存任务，满则丢弃最旧一帧"""
        if not self.is_running:
            return
        try:
            self.save_queue.put_nowait((color_src, depth_src, int(timestamp_ns)))
        except queue.Full:
            try:
                _ = self.save_queue.get_nowait()
                self.save_queue.task_done()
            except Exception:
                pass
            try:
                self.save_queue.put_nowait((color_src, depth_src, int(timestamp_ns)))
            except Exception:
                pass

    def save_and_show(self, img, depth):
        """兼容旧接口：显示+提交保存（数组）"""
        if not self.is_running:
            return
        try:
            if self.display_queue.full():
                self.display_queue.get_nowait()
            timestamp = time.time_ns()
            self.display_queue.put((img, timestamp))
            self.enqueue_save(img, depth, timestamp)
        except Exception as e:
            print(f"显示队列异常: {e}")

    def stop(self):
        """停止显示线程"""
        # 先等待保存队列清空，避免丢帧
        try:
            self.save_queue.join()
        except Exception:
            pass
        # 通知各线程结束
        self.is_running = False
        try:
            self.display_thread.join(timeout=1.0)
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
    # 绑定采集线程（主循环）到读取核
    _set_current_thread_affinity(ORBBEC_READ_CORES_DEFAULT)
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
                # 使用RGB格式，由我们统一转换为BGR
                color_profile = profile_list.get_video_stream_profile(COLOR_WIDTH, COLOR_HEIGHT, OBFormat.RGB, COLOR_FPS)
                print(f"✓ 使用自定义彩色流配置: {COLOR_WIDTH}x{COLOR_HEIGHT} RGB {COLOR_FPS}fps")
            except:
                color_profile = profile_list.get_default_video_stream_profile()
                print("✓ 使用默认彩色流配置")
            config.enable_stream(color_profile)
            
            # 获取深度流配置 - 自定义分辨率（优先 1280x720 -> 回退 1280x800 -> 默认）
            profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = None
            try:
                depth_profile = profile_list.get_video_stream_profile(DEPTH_WIDTH, DEPTH_HEIGHT, OBFormat.Y16, DEPTH_FPS)
                print(f"✓ 使用自定义深度流配置: {DEPTH_WIDTH}x{DEPTH_HEIGHT} Y16 {DEPTH_FPS}fps")
            except Exception:
                try:
                    # 部分设备不支持 1280x720 深度，但支持 1280x800
                    depth_profile = profile_list.get_video_stream_profile(1280, 800, OBFormat.Y16, DEPTH_FPS)
                    print("✓ 使用回退深度流配置: 1280x800 Y16 {fps}fps".format(fps=DEPTH_FPS))
                except Exception:
                    depth_profile = profile_list.get_default_video_stream_profile()
                    print("✓ 使用默认深度流配置")
            config.enable_stream(depth_profile)
            
            # 启用硬件D2C对齐
            config.set_align_mode(OBAlignMode.HW_MODE)
            print("启用硬件D2C对齐")
            
            # 启动pipeline
            pipeline.start(config)
            print("头部奥比中光相机连接成功")
            
            
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
                    
                    # 处理彩色图像用于预览显示
                    color_image = frame_to_bgr_image(aligned_color_frame)
                    if color_image is None:
                        continue

                    # 提交保存任务（传递帧对象，减少主线程拷贝）
                    timestamp_ns = time.time_ns()
                    displayer.enqueue_save(aligned_color_frame, aligned_depth_frame, timestamp_ns)

                    # 实时预览
                    displayer.show_image(color_image)

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
                        print(f"OB获取30张图片耗时: {elapsed_time:.3f}秒, 平均FPS: {fps:.2f}")
                        
                        # 重置计数器，开始下一轮统计
                        start_time_30 = time.time()
                        frame_count_30 = 0
                    
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
            try:
                # 关闭显示与保存线程
                displayer.stop()
            except Exception:
                pass
            try:
                displayer.destroy_node()
            except Exception:
                pass
            try:
                rclpy.shutdown()
            except Exception:
                pass

if __name__ == "__main__":
    current_process = psutil.Process(os.getpid())
    # 进程亲和性设为读取/写入线程核集合的并集，可通过环境变量覆盖
    read_cores_env = _parse_cores_env("ORBBEC_READ_CORES", ORBBEC_READ_CORES_DEFAULT)
    write_cores_env = _parse_cores_env("ORBBEC_WRITE_CORES", ORBBEC_WRITE_CORES_DEFAULT)
    all_cores = sorted(set(read_cores_env + write_cores_env))
    if all_cores:
        current_process.cpu_affinity(all_cores)
    _orbbec_camera_loop_head()
