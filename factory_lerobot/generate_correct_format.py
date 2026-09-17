#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import os
import subprocess
import yaml

# 设置奥比中光SDK路径
ORBBEC_SDK_PATH = "/home/dreame/package/pyorbbecsdk"
sys.path.append(f"{ORBBEC_SDK_PATH}/install/lib")
sys.path.append(f"{ORBBEC_SDK_PATH}/examples")

# 设置环境变量
os.environ['PYTHONPATH'] = f"{ORBBEC_SDK_PATH}/install/lib:{os.environ.get('PYTHONPATH', '')}"
os.environ['LD_LIBRARY_PATH'] = f"{ORBBEC_SDK_PATH}/install/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

try:
    import pyorbbecsdk
    from pyorbbecsdk import *
    print("✓ 奥比中光SDK 可用")
except ImportError as e:
    print(f"✗ 奥比中光SDK不可用: {e}")
    sys.exit(1)

try:
    import pyrealsense2 as rs
    print("✓ pyrealsense2 可用")
except ImportError as e:
    print(f"✗ pyrealsense2不可用: {e}")
    sys.exit(1)

def get_ros_domain_id_from_remote():
    """从远程服务器读取ROS_DOMAIN_ID"""
    try:
        print("正在从远程服务器读取ROS_DOMAIN_ID...")
        # 使用sshpass连接到远程服务器，先source环境文件，再读取ROS_DOMAIN_ID
        cmd = "sshpass -p '123' ssh eame@192.168.54.119 'source /etc/eame/ros2.env && echo $ROS_DOMAIN_ID'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            output = result.stdout.strip()
            # 提取最后一行作为ROS_DOMAIN_ID（过滤掉其他输出信息）
            lines = output.split('\n')
            ros_domain_id = lines[-1].strip()  # 取最后一行
            
            if ros_domain_id and ros_domain_id.isdigit():
                print(f"✓ 从远程服务器读取到ROS_DOMAIN_ID: {ros_domain_id}")
                return ros_domain_id
            else:
                print("✓ 远程服务器上ROS_DOMAIN_ID未设置或格式不正确，使用默认值")
                return "0"
        else:
            print(f"✗ 读取ROS_DOMAIN_ID失败: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("✗ 连接远程服务器超时")
        return None
    except Exception as e:
        print(f"✗ 读取ROS_DOMAIN_ID时出错: {e}")
        return None

def get_dev_sn_from_remote():
    """从远程服务器读取dev_sn.json文件内容"""
    try:
        print("正在从远程服务器读取dev_sn.json...")
        cmd = "sshpass -p '123' ssh eame@192.168.54.119 'cat /home/eame/devID/dev_sn.json'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)

        if result.returncode == 0:
            dev_sn_json = result.stdout.strip()
            if dev_sn_json:
                print(f"✓ 成功读取dev_sn.json文件")
                try:
                    dev_sn_data = json.loads(dev_sn_json)
                    return dev_sn_data
                except json.JSONDecodeError as e:
                    print(f"✗ JSON解析失败: {e}")
                    return dev_sn_json
            else:
                print("✓ dev_sn.json文件为空")
                return {}
        else:
            print(f"✗ 读取dev_sn.json失败: {result.stderr}")
            return {}
    except subprocess.TimeoutExpired:
        print("✗ 连接远程服务器超时")
        return {}
    except Exception as e:
        print(f"✗ 读取dev_sn.json时出错: {e}")
        return {}

def get_robot_parameters_from_remote():
    """从远程服务器读取robot-parameters_n.yaml文件内容"""
    try:
        print("正在从远程服务器读取robot-parameters_n.yaml...")
        # 使用sshpass连接到远程服务器，读取robot-parameters_n.yaml文件内容
        cmd = "sshpass -p '123' ssh eame@192.168.54.110 'cat /usr/dreame_humanoid_robot/robot-parameters_n.yaml'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            robot_params_yaml = result.stdout.strip()
            if robot_params_yaml:
                print(f"✓ 成功读取robot-parameters_n.yaml文件")
                # 将YAML字符串解析为Python对象
                try:
                    robot_params = yaml.safe_load(robot_params_yaml)
                    return robot_params
                except yaml.YAMLError as e:
                    print(f"✗ YAML解析失败: {e}")
                    return robot_params_yaml  # 如果解析失败，返回原始字符串
            else:
                print("✓ robot-parameters_n.yaml文件为空")
                return {}
        else:
            print(f"✗ 读取robot-parameters_n.yaml失败: {result.stderr}")
            return {}
            
    except subprocess.TimeoutExpired:
        print("✗ 连接远程服务器超时")
        return {}
    except Exception as e:
        print(f"✗ 读取robot-parameters_n.yaml时出错: {e}")
        return {}

def get_realsense_info():
    """获取所有RealSense相机信息"""
    # 自动读取ros_node_save_data_orbbec.py中的SN配置
    def read_sn_config():
        try:
            config_file = "/home/dreame/test/factory_lerobot/act_get_data/act_get_data/ros_node_save_data_orbbec.py"
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 使用正则表达式提取SN配置
            import re
            left_sn_match = re.search(r'self\.realsense_camera_sn_l\s*=\s*"([^"]+)"', content)
            right_sn_match = re.search(r'self\.realsense_camera_sn_r\s*=\s*"([^"]+)"', content)
            
            left_sn = left_sn_match.group(1) if left_sn_match else None
            right_sn = right_sn_match.group(1) if right_sn_match else None
            
            print(f"自动读取项目配置:")
            print(f"  左手相机SN: {left_sn}")
            print(f"  右手相机SN: {right_sn}")
            
            return left_sn, right_sn
        except Exception as e:
            print(f"⚠️ 自动读取配置失败: {e}")
            print("无法获取左右手SN配置，将无法准确区分左右手")
            return None, None
    
    LEFT_HAND_SN, RIGHT_HAND_SN = read_sn_config()
    
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        
        cameras = []
        
        print(f"检测到 {len(devices)} 个RealSense设备")
        
        for i, dev in enumerate(devices):
            name = dev.get_info(rs.camera_info.name)
            sn = dev.get_info(rs.camera_info.serial_number)
            firmware = dev.get_info(rs.camera_info.firmware_version)
            usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
            
            print(f"设备 {i+1}: {name} (SN: {sn})")
            
            try:
                # 创建pipeline
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(sn)
                config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
                config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
                
                profile = pipeline.start(config)
                
                # 获取彩色流内参
                color_stream = profile.get_stream(rs.stream.color)
                color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
                
                # 获取深度流内参
                depth_stream = profile.get_stream(rs.stream.depth)
                depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
                
                # 根据官方文档，Brown-Conrady模型顺序为 [k1, k2, p1, p2, k3]
                # RealSense SDK的coeffs数组顺序是 [k1, k2, p1, p2, k3]
                color_coeffs = {
                    "k1": color_intrinsics.coeffs[0],  # coeffs[0] - k1
                    "k2": color_intrinsics.coeffs[1],  # coeffs[1] - k2
                    "p1": color_intrinsics.coeffs[2],  # coeffs[2] - p1
                    "p2": color_intrinsics.coeffs[3],  # coeffs[3] - p2
                    "k3": color_intrinsics.coeffs[4]   # coeffs[4] - k3
                }
                
                depth_coeffs = {
                    "k1": depth_intrinsics.coeffs[0],  # coeffs[0] - k1
                    "k2": depth_intrinsics.coeffs[1],  # coeffs[1] - k2
                    "p1": depth_intrinsics.coeffs[2],  # coeffs[2] - p1
                    "p2": depth_intrinsics.coeffs[3],  # coeffs[3] - p2
                    "k3": depth_intrinsics.coeffs[4]   # coeffs[4] - k3
                }
                
                pipeline.stop()
                
                # 根据项目配置的SN判断左右手
                if LEFT_HAND_SN is None or RIGHT_HAND_SN is None:
                    # 配置读取失败，无法区分左右手
                    hand_position = "config_read_error"
                    print(f"⚠️ 配置读取失败，无法区分左右手 (SN: {sn})")
                elif sn == LEFT_HAND_SN:
                    hand_position = "left_hand"
                    print(f"✓ 识别为左手相机 (SN: {sn})")
                elif sn == RIGHT_HAND_SN:
                    hand_position = "right_hand"
                    print(f"✓ 识别为右手相机 (SN: {sn})")
                else:
                    # SN不匹配时，标记错误但使用识别到的SN
                    hand_position = "error_sn_mismatch"
                    print(f"⚠️ SN不匹配! 识别到: {sn}")
                    print(f"  配置的左手SN: {LEFT_HAND_SN}")
                    print(f"  配置的右手SN: {RIGHT_HAND_SN}")
                    print(f"  使用识别到的SN，hand_position标记为error")
                
                camera_info = {
                    "type": "realsense",
                    "name": name,
                    "serial_number": sn,
                    "firmware_version": firmware,
                    "usb_type": usb_type,
                    "device_index": i,
                    "hand_position": hand_position,
                    "color_intrinsics": {
                        "width": color_intrinsics.width,
                        "height": color_intrinsics.height,
                        "fx": color_intrinsics.fx,
                        "fy": color_intrinsics.fy,
                        "ppx": color_intrinsics.ppx,
                        "ppy": color_intrinsics.ppy,
                        "distortion_model": str(color_intrinsics.model),
                        "distortion_coeffs": color_coeffs
                    },
                    "depth_intrinsics": {
                        "width": depth_intrinsics.width,
                        "height": depth_intrinsics.height,
                        "fx": depth_intrinsics.fx,
                        "fy": depth_intrinsics.fy,
                        "ppx": depth_intrinsics.ppx,
                        "ppy": depth_intrinsics.ppy,
                        "distortion_model": str(depth_intrinsics.model),
                        "distortion_coeffs": depth_coeffs
                    }
                }
                
                cameras.append(camera_info)
                print(f"✓ 成功获取设备 {i+1} 的参数")
                
            except Exception as e:
                print(f"✗ 获取设备 {i+1} 参数失败: {e}")
                continue
        
        return cameras
            
    except Exception as e:
        print(f"获取RealSense信息失败: {e}")
        return []

def get_orbbec_info():
    """获取奥比中光相机信息"""
    try:
        context = Context()
        device_list = context.query_devices()
        
        if device_list.get_count() > 0:
            device = device_list[0]
            device_info = device.get_device_info()
            sn = device_info.get_serial_number()
            name = device_info.get_name()
            
            print(f"设备: {name} (SN: {sn})")
            
            # 获取彩色和深度传感器
            sensor_list = device.get_sensor_list()
            color_sensor = None
            depth_sensor = None
            
            for j in range(sensor_list.get_count()):
                sensor_type = sensor_list.get_type_by_index(j)
                if sensor_type == OBSensorType.COLOR_SENSOR:
                    color_sensor = sensor_list.get_sensor_by_index(j)
                elif sensor_type == OBSensorType.DEPTH_SENSOR:
                    depth_sensor = sensor_list.get_sensor_by_index(j)
            
            camera_info = {
                "type": "orbbec",
                "name": name,
                "serial_number": sn,
                "firmware_version": "unknown",
                "usb_type": "3.0",
                "device_index": 0
            }
            
            # 获取彩色流信息
            if color_sensor:
                color_profile_list = color_sensor.get_stream_profile_list()
                if color_profile_list.get_count() > 0:
                    color_profile = color_profile_list.get_stream_profile_by_index(0)
                    color_video_profile = color_profile.as_video_stream_profile()
                    
                    color_intrinsics = color_video_profile.get_intrinsic()
                    color_distortion = color_video_profile.get_distortion()
                    
                    # 根据奥比中光SDK官方文档，参数顺序为 [k1, k2, k3, k4, k5, k6, p1, p2]
                    color_coeffs = {
                        "k1": color_distortion.k1,
                        "k2": color_distortion.k2,
                        "k3": color_distortion.k3,
                        "k4": color_distortion.k4,
                        "k5": color_distortion.k5,
                        "k6": color_distortion.k6,
                        "p1": color_distortion.p1,
                        "p2": color_distortion.p2
                    }
                    
                    camera_info["color_intrinsics"] = {
                        "width": color_video_profile.get_width(),
                        "height": color_video_profile.get_height(),
                        "fx": color_intrinsics.fx,
                        "fy": color_intrinsics.fy,
                        "ppx": color_intrinsics.cx,
                        "ppy": color_intrinsics.cy,
                        "distortion_model": "brown_conrady",
                        "distortion_coeffs": color_coeffs
                    }
            
            # 获取深度流信息
            if depth_sensor:
                depth_profile_list = depth_sensor.get_stream_profile_list()
                if depth_profile_list.get_count() > 0:
                    depth_profile = depth_profile_list.get_stream_profile_by_index(0)
                    depth_video_profile = depth_profile.as_video_stream_profile()
                    
                    depth_intrinsics = depth_video_profile.get_intrinsic()
                    depth_distortion = depth_video_profile.get_distortion()
                    
                    # 根据奥比中光SDK官方文档，参数顺序为 [k1, k2, k3, k4, k5, k6, p1, p2]
                    depth_coeffs = {
                        "k1": depth_distortion.k1,
                        "k2": depth_distortion.k2,
                        "k3": depth_distortion.k3,
                        "k4": depth_distortion.k4,
                        "k5": depth_distortion.k5,
                        "k6": depth_distortion.k6,
                        "p1": depth_distortion.p1,
                        "p2": depth_distortion.p2
                    }
                    
                    camera_info["depth_intrinsics"] = {
                        "width": depth_video_profile.get_width(),
                        "height": depth_video_profile.get_height(),
                        "fx": depth_intrinsics.fx,
                        "fy": depth_intrinsics.fy,
                        "ppx": depth_intrinsics.cx,
                        "ppy": depth_intrinsics.cy,
                        "distortion_model": "brown_conrady",
                        "distortion_coeffs": depth_coeffs
                    }
                    
                    return camera_info
                    
    except Exception as e:
        print(f"获取奥比中光信息失败: {e}")
        return None

if __name__ == "__main__":
    # 获取dev_sn.json内容
    dev_sn = get_dev_sn_from_remote()
    
    # 获取ROS_DOMAIN_ID
    ros_domain_id = get_ros_domain_id_from_remote()
    if ros_domain_id is None:
        print("使用默认ROS_DOMAIN_ID: 0")
        ros_domain_id = "0"
    
    # 获取robot-parameters_n.yaml内容
    robot_parameters = get_robot_parameters_from_remote()
    
    cameras = []
    
    print("\n=== RealSense相机信息 ===")
    realsense_info = get_realsense_info()
    if realsense_info:
        cameras.extend(realsense_info)
    
    print("\n=== 奥比中光相机信息 ===")
    orbbec_info = get_orbbec_info()
    if orbbec_info:
        cameras.append(orbbec_info)
    
    # 创建包含dev_sn、ROS_DOMAIN_ID和robot-parameters的完整数据结构
    # 如果dev_sn是字典且包含dev_sn字段，直接使用其值
    if isinstance(dev_sn, dict) and "dev_sn" in dev_sn:
        dev_sn_value = dev_sn["dev_sn"]
    else:
        dev_sn_value = dev_sn
    
    complete_data = {
        "dev_sn": dev_sn_value,
        "ros_domain_id": ros_domain_id,
        "robot_parameters": robot_parameters,
        "cameras": cameras
    }
    
    # 保存为JSON文件，使用ROS_DOMAIN_ID作为文件名
    output_file = f"info_domain_id_{ros_domain_id}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(complete_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n相机信息已保存到: {output_file}")
    
    # 显示JSON内容
    print(f"\n=== JSON内容 ===")
    print(json.dumps(complete_data, indent=2, ensure_ascii=False)) 
