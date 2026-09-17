#!/usr/bin/env python3
"""
奥比中光多线程RGB+深度图像采集程序
基于sync_align.py，类似RealSense的多线程结构
"""

import cv2
import numpy as np
import time
import threading
import copy
from pyorbbecsdk import *
import sys
import os
sys.path.append('/home/dreame/package/pyorbbecsdk/examples')
from utils import frame_to_bgr_image

class OrbbecMultiThreadCapture:
    def __init__(self):
        self.running = True
        self.pipeline = None
        self.align_filter = None
        
        # 数据存储
        self.color_image = None
        self.depth_data = None
        self.timestamp = None
        
        # 线程锁
        self.data_lock = threading.Lock()
        
        # 线程
        self.capture_thread = None
        
    def connect_camera(self):
        """连接相机"""
        try:
            self.pipeline = Pipeline()
            config = Config()
            
            # 获取彩色流配置
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(color_profile)
            
            # 获取深度流配置
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(depth_profile)
            
            # 启动pipeline
            self.pipeline.start(config)
            print("✓ 奥比中光相机连接成功")
            
            # 创建对齐过滤器
            self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
            
            return True
            
        except Exception as e:
            print(f"✗ 相机连接失败: {e}")
            return False
    
    def _capture_loop(self):
        """采集循环（在独立线程中运行）"""
        print("开始图像采集循环...")
        
        while self.running:
            try:
                # 等待帧数据
                frames = self.pipeline.wait_for_frames(1000)
                if not frames:
                    continue
                
                # 获取原始帧
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # 应用对齐过滤器
                aligned_frames = self.align_filter.process(frames)
                if not aligned_frames:
                    continue
                
                aligned_frames = aligned_frames.as_frame_set()
                aligned_color_frame = aligned_frames.get_color_frame()
                aligned_depth_frame = aligned_frames.get_depth_frame()
                
                if not aligned_color_frame or not aligned_depth_frame:
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
                
                # 深度数据处理
                depth_data = depth_data.astype(np.float32) * aligned_depth_frame.get_depth_scale()
                depth_data = np.where((depth_data > 20) & (depth_data < 10000), depth_data, 0)
                depth_data = depth_data.astype(np.uint16)
                
                # 深度图像翻转
                # depth_data = cv2.flip(depth_data, -1)  # -1表示水平和垂直翻转
                
                # 线程安全地更新数据
                with self.data_lock:
                    self.color_image = copy.copy(color_image)
                    self.depth_data = copy.copy(depth_data)
                    self.timestamp = time.time()
                
                # 控制帧率
                time.sleep(0.001)
                
            except Exception as e:
                # 设备中断时抛出异常（类似RealSense）
                if "Frame didn't arrive" in str(e) or "timeout" in str(e).lower():
                    print(f"检测到设备连接中断: {e}")
                    break
                else:
                    print(f"采集错误: {e}")
                    continue
    
    def start_capture(self):
        """启动采集线程"""
        if not self.connect_camera():
            return False
        
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print("✓ 采集线程启动成功")
        return True
    
    def stop_capture(self):
        """停止采集"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        
        if self.pipeline:
            self.pipeline.stop()
        print("✓ 采集已停止")
    
    def get_latest_data(self):
        """获取最新数据（线程安全）"""
        with self.data_lock:
            return {
                'color_image': copy.copy(self.color_image) if self.color_image is not None else None,
                'depth_data': copy.copy(self.depth_data) if self.depth_data is not None else None,
                'timestamp': self.timestamp
            }
    
    def save_image(self, color_path, depth_path):
        """保存图像"""
        data = self.get_latest_data()
        if data['color_image'] is not None:
            cv2.imwrite(color_path, data['color_image'])
            print(f"✓ RGB图像已保存: {color_path}")
        
        if data['depth_data'] is not None:
            # 保存原始深度数据为numpy格式
            np.save(depth_path.replace('.png', '.npy'), data['depth_data'])
            # 保存原始深度图（转换为可显示格式）
            depth_display = cv2.convertScaleAbs(data['depth_data'], alpha=0.03)
            cv2.imwrite(depth_path, depth_display)
            print(f"✓ 深度图像已保存: {depth_path}")

def main():
    """主函数"""
    print("=== 奥比中光多线程RGB+深度采集程序 ===")
    
    # 创建采集对象
    capture = OrbbecMultiThreadCapture()
    
    try:
        # 启动采集
        if not capture.start_capture():
            print("启动失败，程序退出")
            return
        
        print("程序运行中...")
        print("按 'q' 退出，按 's' 保存图像")
        
        # 主循环
        while True:
            # 获取最新数据
            data = capture.get_latest_data()
            
            # 显示图像
            if data['color_image'] is not None:
                cv2.imshow("Orbbec RGB", data['color_image'])
            
            if data['depth_data'] is not None:
                # 显示原始深度图（转换为可显示格式）
                depth_display = cv2.convertScaleAbs(data['depth_data'], alpha=0.03)
                cv2.imshow("Orbbec Depth", depth_display)
            
            # 键盘处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存图像
                timestamp = int(time.time() * 1000)
                capture.save_image(
                    f"orbbec_rgb_{timestamp}.png",
                    f"orbbec_depth_{timestamp}.png"
                )
        
    except KeyboardInterrupt:
        print("\n收到中断信号")
    except Exception as e:
        print(f"程序错误: {e}")
    finally:
        # 清理资源
        capture.stop_capture()
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == "__main__":
    main() 