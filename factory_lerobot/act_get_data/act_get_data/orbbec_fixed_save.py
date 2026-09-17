#!/usr/bin/env python3

import cv2
import numpy as np
import time
import os
import sys
import threading
import queue
from datetime import datetime
import psutil
import ctypes
from pyorbbecsdk import *

# 添加robot_kinemic模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'robot_kinemic', 'robot_kinemic'))
from utils import frame_to_bgr_image

class AsyncImageDisplayer:
    def __init__(self, target_fps=30, save_dir="/tmp/orbbec_images"):
        # 基本设置
        self.target_fps = target_fps
        self.min_interval = 1.0 / target_fps
        self.last_display_time = 0.0
        self.window_name = "head Display"
        self.is_running = True
        
        # 保存相关
        self.save_dir = save_dir
        self.frame_count = 0
        self.setup_save_directory()
        
        # 队列
        self.display_queue = queue.Queue(maxsize=1)
        self.save_queue = queue.Queue(maxsize=1)
        
        # 启动线程
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        self.save_thread = threading.Thread(target=self._save_loop, daemon=True)
        self.save_thread.start()

    def setup_save_directory(self):
        """创建保存目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.save_dir, f"session_{timestamp}")
        self.rgb_dir = os.path.join(self.session_dir, "rgb")
        self.depth_dir = os.path.join(self.session_dir, "depth")
        
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        
        self.timestamp_file = os.path.join(self.session_dir, "timestamps.txt")
        print(f"保存目录: {self.session_dir}")

    def _resize_for_display(self, img):
        return cv2.resize(img, (640, 360))

    def _display_loop(self):
        """显示线程主循环"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        while self.is_running:
            try:
                # 带超时的get，避免永久阻塞
                img, timestamp = self.display_queue.get(timeout=0.1)
                
                # 检查是否达到显示频率要求
                current_time = time.time()
                if current_time - self.last_display_time >= self.min_interval:
                    # 缩放图像并显示
                    display_img = self._resize_for_display(img)
                    cv2.imshow(self.window_name, display_img)
                    self.last_display_time = current_time
                    
                    # 处理窗口事件
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.stop()
                
                self.display_queue.task_done()
                
            except queue.Empty:
                continue

    def _save_loop(self):
        """保存线程主循环"""
        while self.is_running:
            try:
                # 带超时的get，避免永久阻塞
                image_color, image_depth, rgb_timestamp = self.save_queue.get(timeout=0.1)

                # 保存图像
                frame_count = self.frame_count
                
                # 保存RGB和深度图像
                image_color_path = os.path.join(self.rgb_dir, f"{frame_count}.jpg")
                image_depth_path = os.path.join(self.depth_dir, f"{frame_count}.png")

                cv2.imwrite(image_color_path, image_color)
                cv2.imwrite(image_depth_path, image_depth)
                
                # 保存时间戳到txt文件
                with open(self.timestamp_file, 'a') as f:
                    f.write(f"{rgb_timestamp}\n")
                
                self.frame_count += 1
                
                # 定期报告进度
                if frame_count % 30 == 0:
                    print(f"已保存 {frame_count} 帧")
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"save failed: {e}")

    def show_image(self, img):
        """向队列中添加待显示的图像"""
        if not self.is_running:
            return
        try:
            if self.display_queue.full():
                self.display_queue.get_nowait()
            self.display_queue.put((img, time.time()))
        except Exception as e:
            print(f"显示队列异常: {e}")

    def save_and_show(self, img, depth):
        if not self.is_running:
            return
        try:
            if self.display_queue.full():
                self.display_queue.get_nowait()

            timestamp = time.time_ns()
            self.display_queue.put((img, timestamp))
            
            if self.save_queue.full():
                self.save_queue.get_nowait()
            self.save_queue.put((img, depth, timestamp))

        except Exception as e:
            print(f"显示队列异常: {e}")

    def stop(self):
        """停止显示线程"""
        self.is_running = False
        self.display_thread.join(timeout=1.0)
        cv2.destroyAllWindows()

def _orbbec_camera_loop_head():
    """头部奥比中光相机采集循环"""
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
    
    cnt = 0
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
            
            # === 图像采集循环 ===
            while displayer.is_running:
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
                    
                    # 使用硬件D2C对齐：直接使用获取的帧
                    aligned_color_frame = color_frame
                    aligned_depth_frame = depth_frame
                    
                    # 检查对齐后的帧
                    if not aligned_color_frame:
                        continue
                    
                    # 处理彩色图像
                    color_image = frame_to_bgr_image(aligned_color_frame)
                    if color_image is None:
                        continue
                    
                    # 处理深度数据
                    try:
                        depth_data = np.frombuffer(aligned_depth_frame.get_data(), dtype=np.uint16).reshape(
                            (aligned_depth_frame.get_height(), aligned_depth_frame.get_width()))
                    except ValueError:
                        continue
                    
                    # 深度数据处理
                    depth_data = depth_data.astype(np.float32) * aligned_depth_frame.get_depth_scale()
                    depth_data = np.where((depth_data > 20) & (depth_data < 10000), depth_data, 0)
                    depth_data = depth_data.astype(np.uint16)
                    
                    displayer.save_and_show(color_image, depth_data)

                    # 30帧时间统计
                    if start_time_30 is None:
                        start_time_30 = time.time()
                        frame_count_30 = 0
                        print("开始30帧时间统计...")
                    
                    frame_count_30 += 1
                    
                    if frame_count_30 >= 300:
                        end_time_30 = time.time()
                        elapsed_time = end_time_30 - start_time_30
                        fps = 300 / elapsed_time
                        print(f"OB获取300张图片耗时: {elapsed_time:.3f}秒, 平均FPS: {fps:.2f}")
                        
                        # 重置计数器
                        start_time_30 = time.time()
                        frame_count_30 = 0

                    cnt+=1
                    save_count+=1
                    if cnt>=30:
                        print(f"head 30fps time use: {time_use - time.time()}, save count: {save_count}")
                        time_use = time.time()
                        cnt = 0
                    time.sleep(0.01)
                    
                except Exception as e:
                    if "Frame didn't arrive" in str(e) or "timeout" in str(e).lower():
                        print(f"检测到设备连接中断: {e}")
                        break
                    else:
                        print(f"采集错误: {e}")
                        continue
                
        except (KeyboardInterrupt, SystemExit):
            print("程序退出...")
            break
            
        except Exception as e:
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
    target_cores = [12,13,14]
    current_process.cpu_affinity(target_cores)
    _orbbec_camera_loop_head()
