#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import threading
import queue
import psutil
from pathlib import Path
from conf import REALSENSE_LEFT_SN
import ctypes

def _set_current_thread_affinity(cores):
    try:
        if not cores:
            return
        tid = threading.get_native_id()
        os.sched_setaffinity(tid, set(cores))
    except Exception:
        pass

# 线程亲和性默认列表（可按需修改）
RS_LEFT_READ_CORES_DEFAULT = [0,1,2,3,12]
RS_LEFT_WRITE_CORES_DEFAULT = [13,14,15]

def _parse_cores_env(var_name: str, default_list):
    val = os.getenv(var_name)
    if not val:
        return default_list
    try:
        parts = [int(x) for x in val.replace(',', ' ').split() if x.strip().isdigit()]
        return parts if parts else default_list
    except Exception:
        return default_list

# 允许通过环境变量覆盖线程亲和核
RS_LEFT_READ_CORES = _parse_cores_env('RS_LEFT_READ_CORES', RS_LEFT_READ_CORES_DEFAULT)
RS_LEFT_WRITE_CORES = _parse_cores_env('RS_LEFT_WRITE_CORES', RS_LEFT_WRITE_CORES_DEFAULT)


# ROS2导入
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

def wait_for_sync_signal(sync_dir, timeout=300):
    """等待同步采集信号"""
    sync_file = Path(sync_dir) / 'start_capture.signal'
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if sync_file.exists():
            try:
                timestamp_ns = int(sync_file.read_text().strip())
                return timestamp_ns
            except:
                pass
        time.sleep(0.001)
    return None

def save_images_async(color_image, depth_image, color_path, depth_path, timestamp_ns, timestamp_file):
    """异步保存图像和时间戳"""
    try:
        cv2.imwrite(color_path, color_image)
        # np.save(depth_path, depth_image)
        cv2.imwrite(depth_path, depth_image)
        
        # 保存时间戳到txt文件
        with open(timestamp_file, 'a') as f:
            f.write(f"{timestamp_ns}\n")
    except Exception as e:
        print(f"RealSense Left保存失败: {e}")

class SaveWorker:
    """固定工作线程+有界队列的保存器"""
    def __init__(self, maxsize=8, num_workers=1, name="Left"):
        self.queue = queue.Queue(maxsize=maxsize)
        self.is_running = True
        self.threads = []
        self.name = name
        self.dropped_count = 0
        # 预分配缓冲，首次保存时按尺寸初始化并复用
        self._color_buf = None
        self._depth_buf = None
        for _ in range(num_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self.threads.append(t)

    def _worker_loop(self):
        # 绑定保存线程亲和核（仅在线程启动时调用一次）
        _set_current_thread_affinity(RS_LEFT_WRITE_CORES)
        while self.is_running:
            try:
                item = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                self.queue.task_done()
                break
            color_src, depth_src, color_path, depth_path, timestamp_ns, timestamp_file = item
            try:
                # 在保存线程中转换，降低主线程拷贝
                if hasattr(color_src, 'get_data'):
                    color_image = np.asanyarray(color_src.get_data())
                else:
                    color_image = color_src
                if hasattr(depth_src, 'get_data'):
                    depth_image = np.asanyarray(depth_src.get_data())
                else:
                    depth_image = depth_src

                # 复用缓冲区写盘，减少临时分配
                if self._color_buf is None or self._color_buf.shape != color_image.shape or self._color_buf.dtype != color_image.dtype:
                    self._color_buf = np.empty_like(color_image)
                np.copyto(self._color_buf, color_image)
                cv2.imwrite(color_path, self._color_buf)

                if self._depth_buf is None or self._depth_buf.shape != depth_image.shape or self._depth_buf.dtype != depth_image.dtype:
                    self._depth_buf = np.empty_like(depth_image)
                np.copyto(self._depth_buf, depth_image)
                cv2.imwrite(depth_path, self._depth_buf)
                with open(timestamp_file, 'a') as f:
                    f.write(f"{timestamp_ns}\n")
            except Exception as e:
                print(f"RealSense Left保存失败: {e}")
            finally:
                self.queue.task_done()

    def submit(self, color_image, depth_image, color_path, depth_path, timestamp_ns, timestamp_file):
        # 非阻塞提交：满则丢弃最旧，避免主线程阻塞与内存增长
        if not self.is_running:
            return
        try:
            self.queue.put_nowait((color_image, depth_image, color_path, depth_path, timestamp_ns, timestamp_file))
        except queue.Full:
            dropped = False
            try:
                self.queue.get_nowait()
                dropped = True
            except Exception:
                pass
            try:
                self.queue.put_nowait((color_image, depth_image, color_path, depth_path, timestamp_ns, timestamp_file))
            except Exception:
                pass
            if dropped:
                self.dropped_count += 1
                print(f"RealSense {self.name}保存队列已满，丢弃1帧，累计丢弃: {self.dropped_count}")

    def drain_and_stop(self):
        # 等待队列清空
        try:
            self.queue.join()
        except Exception:
            pass
        self.is_running = False
        # 通知线程退出
        for _ in self.threads:
            try:
                self.queue.put_nowait(None)
            except Exception:
                pass
        for t in self.threads:
            t.join(timeout=2.0)

class RealsenseLeftRosController(Node):
    """RealSense Left相机ROS2控制节点"""
    def __init__(self):
        super().__init__('realsense_left_controller')
        
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
            '/realsense_left_control',
            self.camera_control_callback,
            10
        )
        
        self.get_logger().info('RealSense Left ROS2控制节点已启动')
    
    def camera_control_callback(self, msg):
        """处理相机控制指令"""
        command = msg.data
        self.get_logger().info(f'RealSense Left收到指令: {command}')
        
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
        #     self.get_logger().warning("RealSense Left已在采集中")
        #     return
        
        self.current_session = session_id
        self.start_timestamp = start_timestamp or time.time()
        self.target_frames = target_frames or 0
        self.is_collecting = True
        self.should_stop = False
        
        if start_timestamp:
            delay = start_timestamp - time.time()
            self.get_logger().info(f"RealSense Left准备采集，会话: {session_id}")
            self.get_logger().info(f"等待{delay:.1f}秒后开始，目标帧数: {target_frames}")
        else:
            self.get_logger().info(f"RealSense Left开始采集，会话: {session_id}")
    
    def stop_collection(self):
        """停止采集"""
        if not self.is_collecting:
            self.get_logger().warning("RealSense Left未在采集中")
            return
        
        self.is_collecting = False
        self.should_stop = True
        self.get_logger().info("RealSense Left停止采集")

def main():
    print("启动RealSense Left ROS2相机节点...")
    
    # 初始化ROS2
    rclpy.init()
    
    # 创建ROS2控制节点
    ros_controller = RealsenseLeftRosController()
    
    # 输出配置（已移除兜底超时停止）
    
    # RealSense Left SN
    
    target_sn = REALSENSE_LEFT_SN
    
    # CPU亲和性设置（允许环境变量覆盖）
    current_process = psutil.Process(os.getpid())
    _all_cores = sorted(set(RS_LEFT_READ_CORES + RS_LEFT_WRITE_CORES))
    if _all_cores:
        try:
            current_process.cpu_affinity(_all_cores)
        except Exception:
            pass
    # 可选：提高进程与I/O优先级（低配机更稳定）
    try:
        current_process.nice(-5)
        if hasattr(current_process, 'ionice'):
            current_process.ionice(psutil.IOPRIO_CLASS_BE, value=0)
    except Exception:
        pass
    
    pipeline = None
    # 可通过环境变量 RS_LEFT_SAVE_QUEUE 配置保存队列大小
    _qsz_env = os.getenv('RS_LEFT_SAVE_QUEUE')
    try:
        _qsz = int(_qsz_env) if _qsz_env is not None else 512
    except Exception:
        _qsz = 512
    save_worker = SaveWorker(maxsize=_qsz, num_workers=1, name="Left")
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    
    try:
        # 绑定主线程到读取核（采集/显示所在线程）
        _set_current_thread_affinity(RS_LEFT_READ_CORES)

        # === RealSense相机初始化 ===
        print("初始化RealSense Left相机...")
        
        # 检查设备
        ctx = rs.context()
        devices = ctx.query_devices()
        
        device_found = False
        for dev in devices:
            if target_sn == dev.get_info(rs.camera_info.serial_number):
                device_found = True
                print(f"找到RealSense Left设备: SN={target_sn}")
                break
        
        if not device_found:
            print(f"未找到RealSense Left设备: SN={target_sn}")
            return
        
        # 配置pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        
        # SN绑定
        config.enable_device(target_sn)
        
        # 流配置
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        
        # 启动pipeline
        pipeline_profile = pipeline.start(config)
        actual_sn = pipeline_profile.get_device().get_info(rs.camera_info.serial_number)
        print(f"RealSense Left连接成功: SN={actual_sn}")
        
        # 创建对齐对象
        align = rs.align(rs.stream.color)
        
        # === ROS2控制循环 ===
        print("等待ROS2控制指令...")
        print("启动实时预览...")
        
        frame_count = 0
        last_session = None
        realsense_dir = None
        rgb_dir = None
        depth_dir = None
        timestamp_file = None
        frame_count_30 = 0
        start_time_30 = None
        # 创建预览窗口（始终显示）
        window_name = "RealSense_Left_Preview"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 360)
        
        while rclpy.ok():
            # 处理ROS2消息
            rclpy.spin_once(ros_controller, timeout_sec=0.01)
            
            # === 获取帧数据 (统一获取，避免重复) ===
            frames = None
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)  # 统一获取一次
            except Exception as e:
                # 获取帧失败，继续下一轮
                continue
            
        # === 实时预览显示
            if frames:
                try:
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        color_image = np.asanyarray(color_frame.get_data())
                    display_img = color_image  # 如需缩放，可改为最近邻
                    # display_img = cv2.resize(color_image, (640, 360), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow(window_name, display_img)
                        
                except Exception as e:
                    # 预览错误不应终止主循环
                    pass
            if start_time_30 is None:
                start_time_30 = time.time()
                frame_count_30 = 0
                print("开始30帧时间统计...")
            frame_count_30 += 1
            if frame_count_30 >= 30:
                end_time_30 = time.time()
                elapsed_time = end_time_30 - start_time_30
                fps = 30 / elapsed_time
                print(f"L获取30张图片耗时: {elapsed_time:.3f}秒, 平均FPS: {fps:.2f}")
                start_time_30 = time.time()
                frame_count_30 = 0
            # 检查退出键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("用户按下q键，退出RealSense Left节点")
                break
            
            # 检查是否需要开始新的采集会话
            if ros_controller.is_collecting and ros_controller.current_session != last_session:
                # 创建新会话的输出目录 - 使用统一的images文件夹结构
                last_session = ros_controller.current_session
                
                # 使用与头部摄像头相同的路径结构
                from conf import REPO_ID
                dataset_path = f"/home/dreame/data/hf_dataset/{REPO_ID}"
                
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
                        episode_folder = Path(dataset_path) / "chunk_000" / f"episode_{latest_episode_num:06d}"
                    else:
                        episode_folder = Path(dataset_path) / "chunk_000" / "episode_000000"
                else:
                    episode_folder = Path(dataset_path) / "chunk_000" / "episode_000000"
                
                # 创建images子文件夹结构
                images_folder = episode_folder / "images"
                rgb_dir = images_folder / "hand_l_rgb"
                depth_dir = images_folder / "hand_l_depth"

            
                
                # 创建images子文件夹结构
                rgb_dir.mkdir(parents=True, exist_ok=True)
                depth_dir.mkdir(parents=True, exist_ok=True)
                
                # 创建时间戳文件 - 放在对应的图像文件夹里
                timestamp_file = rgb_dir / "hand_l_timestamps.txt"
                # 清空之前的时间戳文件
                if timestamp_file.exists():
                    timestamp_file.unlink()
                
                frame_count = 0
                print(f"RealSense Left使用episode目录: {episode_folder}")
                print(f"RealSense Left时间戳文件: {timestamp_file}")
            
            # 采集处理（如果正在采集且有frames）
            if ros_controller.is_collecting and not ros_controller.should_stop and frames:
                try:
                    current_time = time.time()
                    
                    # 如果设置了开始时间戳，等待到指定时间
                    if ros_controller.start_timestamp and current_time < ros_controller.start_timestamp:
                        remaining_wait = ros_controller.start_timestamp - current_time
                        if frame_count == 0:  # 只在第一次显示等待信息
                            print(f"RealSense Left等待{remaining_wait:.1f}秒后开始采集...")
                        if os.path.exists(timestamp_file):
                            os.remove(timestamp_file)
                        continue
                    
                    # 第一次开始采集时记录实际开始时间
                    if frame_count == 0:
                        ros_controller.collection_start_time = current_time
                        print(f"RealSense Left开始采集，目标帧数: {ros_controller.target_frames}")
                    
                    # 检查是否达到目标帧数
                    if ros_controller.target_frames > 0 and frame_count >= ros_controller.target_frames:
                        print(f"RealSense Left达到目标帧数{ros_controller.target_frames}，自动停止")
                        ros_controller.stop_collection()
                        continue
                    
                    # 无兜底超时停止逻辑
                    
                    # 对齐帧
                    aligned_frames = align.process(frames)
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                    
                    if not color_frame or not depth_frame:
                        continue
                    
                    # 使用简单数字命名保存图像
                    timestamp_ns = time.time_ns()
                    rgb_path = rgb_dir / f"{frame_count}.jpg"
                    depth_path = depth_dir / f"{frame_count}.png"
                    
                    # 提交到固定保存线程（包含时间戳），仅传递frame对象
                    save_worker.submit(color_frame, depth_frame, str(rgb_path), str(depth_path), timestamp_ns, str(timestamp_file))
                    
                    frame_count += 1
                    
                    # 定期报告进度
                    if frame_count % 30 == 0 and ros_controller.collection_start_time:
                        elapsed = time.time() - ros_controller.collection_start_time
                        fps = frame_count / elapsed if elapsed > 0 else 0
                        
                        if ros_controller.target_frames > 0:
                            progress = (frame_count / ros_controller.target_frames) * 100
                            remaining_frames = ros_controller.target_frames - frame_count
                            print(f"RealSense Left: {frame_count}/{ros_controller.target_frames}帧 ({progress:.1f}%), {fps:.1f}fps, 剩余{remaining_frames}帧")
                        else:
                            print(f"RealSense Left: {frame_count}帧, {fps:.1f}fps")
                
                except Exception as e:
                    print(f"RealSense Left采集错误: {e}")
                    continue
            
            elif ros_controller.should_stop:
                # 完成当前会话的清理
                elapsed = time.time() - ros_controller.collection_start_time if ros_controller.collection_start_time else 0
                print(f"RealSense Left采集完成: {frame_count}帧，用时{elapsed:.1f}秒")
                print(f"RealSense Left时间戳已保存到: {timestamp_file}")
                ros_controller.should_stop = False
                frame_count = 0

        
    except Exception as e:
        print(f"RealSense Left初始化失败: {e}")
        return
    
    finally:
        # 清理资源
        if pipeline:
            try:
                pipeline.stop()
            except:
                pass
        
        cv2.destroyAllWindows()
        
        # 等待保存线程完成
        print("RealSense Left等待保存完成...")
        try:
            save_worker.drain_and_stop()
        except Exception:
            pass
        
        # 清理ROS2资源
        try:
            ros_controller.destroy_node()
            rclpy.shutdown()
        except:
            pass
        
        print("RealSense Left ROS2相机节点停止")

if __name__ == "__main__":
    main()
