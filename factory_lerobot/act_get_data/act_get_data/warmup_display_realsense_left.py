#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealSense Left预热显示脚本 - Step命名版本
1. 初始化相机并显示实时画面
2. 预热阶段：只显示，不保存
3. 等待用户信号后开始同步采集
4. 图片使用step命名，时间戳保存到txt文件
"""

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

# ROS2导入
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# 线程绑核（读取/保存分别绑定）
RS_LEFT_READ_CORES = [0,1,2,3,12]
RS_LEFT_WRITE_CORES = [13,14,15]

def _bind_current_thread(cores):
    try:
        if not cores:
            return
        tid = threading.get_native_id()
        os.sched_setaffinity(tid, set(cores))
    except Exception:
        pass

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
    # 保存线程绑定到写入核
    _bind_current_thread(RS_LEFT_WRITE_CORES)
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
    def __init__(self, maxsize=2048):
        self.queue = queue.Queue(maxsize=maxsize)
        self.is_running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.dropped_count = 0
        self.thread.start()

    def _worker_loop(self):
        _bind_current_thread(RS_LEFT_WRITE_CORES)
        while self.is_running:
            try:
                item = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                self.queue.task_done()
                break
            color_image, depth_image, color_path, depth_path, timestamp_ns, timestamp_file = item
            try:
                cv2.imwrite(color_path, color_image)
                cv2.imwrite(depth_path, depth_image)
                with open(timestamp_file, 'a') as f:
                    f.write(f"{timestamp_ns}\n")
            except Exception as e:
                print(f"RealSense Left保存失败: {e}")
            finally:
                # 释放引用，便于GC
                del color_image
                del depth_image
                self.queue.task_done()

    def submit(self, color_image, depth_image, color_path, depth_path, timestamp_ns, timestamp_file):
        if not self.is_running:
            return
        try:
            self.queue.put_nowait((color_image, depth_image, color_path, depth_path, timestamp_ns, timestamp_file))
        except queue.Full:
            # 丢弃最旧，保证采集不中断
            try:
                _ = self.queue.get_nowait()
                self.queue.task_done()
            except Exception:
                pass
            try:
                self.queue.put_nowait((color_image, depth_image, color_path, depth_path, timestamp_ns, timestamp_file))
            except Exception:
                self.dropped_count += 1

    def drain_and_stop(self):
        try:
            self.queue.join()
        except Exception:
            pass
        self.is_running = False
        try:
            self.queue.put_nowait(None)
        except Exception:
            pass
        self.thread.join(timeout=2.0)

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
    
    # 输出配置
    output_root = "/home/dreame/lhl/camera_data"
    duration = 60.0  # 默认采集60秒
    
    # RealSense Left SN
    
    target_sn = REALSENSE_LEFT_SN
    
    # CPU亲和性设置
    # 仅线程绑核，不设进程亲和
    
    pipeline = None
    save_worker = None
    
    try:
        # 绑定主线程到读取核（仅一次）
        _bind_current_thread(RS_LEFT_READ_CORES)

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
        
        # 初始化保存队列工作线程（队列大）
        try:
            qsz_env = os.getenv('RS_LEFT_SAVE_QUEUE')
            qsz = int(qsz_env) if qsz_env is not None else 4096
        except Exception:
            qsz = 4096
        save_worker = SaveWorker(maxsize=qsz)

        # === ROS2控制循环 ===
        print("等待ROS2控制指令...")
        print("启动实时预览...")
        
        frame_count = 0
        last_session = None
        realsense_dir = None
        rgb_dir = None
        depth_dir = None
        timestamp_file = None
        
        # 创建预览窗口
        window_name = "RealSense_Left_Preview"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 360)
        
        display_toggle = 0  # 15Hz显示：每2帧显示1帧

        while rclpy.ok():
            # 处理ROS2消息
            rclpy.spin_once(ros_controller, timeout_sec=0.01)
            
            # === 获取帧数据 (统一获取，避免重复) ===
            frames = None
            try:
                frames = pipeline.wait_for_frames(timeout_ms=100)  # 统一获取一次
            except Exception as e:
                # 获取帧失败，继续下一轮
                continue
            
            # === 实时预览显示 (始终显示) ===
            if frames:
                try:
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        color_image = np.asanyarray(color_frame.get_data())
                        
                        # 显示预览画面（15Hz：每2帧显示1帧）
                        display_toggle ^= 1
                        if display_toggle == 0:
                            display_img = color_image  # 如需缩放，可改为最近邻
                            # display_img = cv2.resize(color_image, (640, 360), interpolation=cv2.INTER_NEAREST)
                            cv2.imshow(window_name, display_img)
                        
                except Exception as e:
                    # 预览错误不影响主循环
                    pass
            
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
                    
                    # 备用超时保护（如果没有设置目标帧数）
                    if ros_controller.target_frames == 0 and ros_controller.collection_start_time:
                        if time.time() - ros_controller.collection_start_time >= duration:
                            print(f"RealSense Left采集时间到达{duration}秒，自动停止")
                            ros_controller.stop_collection()
                            continue
                    
                    # 对齐帧
                    aligned_frames = align.process(frames)
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                    
                    if not color_frame or not depth_frame:
                        continue
                    
                    # 转换为numpy数组
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    
                    
                    # 使用简单数字命名保存图像
                    timestamp_ns = time.time_ns()
                    rgb_path = rgb_dir / f"{frame_count}.jpg"
                    depth_path = depth_dir / f"{frame_count}.png"
                    
                    # 提交到保存队列，非阻塞
                    save_worker.submit(color_image, depth_image, str(rgb_path), str(depth_path), timestamp_ns, str(timestamp_file))
                    
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
                            remaining = duration - elapsed
                            print(f"RealSense Left: {frame_count}帧, {fps:.1f}fps, 剩余{remaining:.1f}s")
                
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
        
        # 等待保存队列完成
        if save_worker:
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
