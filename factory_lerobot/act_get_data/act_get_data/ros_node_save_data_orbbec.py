import os
import time
import numpy as np
import copy
import wave
import cv2
from conf import *
from lcm_unit import lcmUnit
from lerobot_unit import lerobotUnit
import torch

import rclpy
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray, String, Int32, Float32MultiArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from voice_msgs.srv import Config
import json
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import queue
import threading
import copy
import shutil
from cv_bridge import CvBridge
from multi_camera_msgs.msg import SyncImg

class VoiceConfigClient:
    def __init__(self, node):
        self.node = node
        self.client = self.node.create_client(Config, '/voice_config')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('语音配置服务不可用，等待中...')

    def send_config(self, enable_voice=True):
        req = Config.Request()
        req.enable_voice = enable_voice
        
        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        
        if future.result() is not None:
            response = future.result()
            return response.result, response.err_str
        else:
            error_msg = future.exception() if future.exception() else "未知错误"
            raise RuntimeError(f'服务调用失败: {error_msg}')

class AudioDataCollector:
    def __init__(self, node):
        self.node = node
        self.bf_audio_queue = queue.Queue(maxsize=1000)      # 波束形成音频队列
        
        # 音频数据缓存
        self.bf_audio_buffer = []
        
        # 音频数据统计
        self.frame_count = 0  # 音频帧计数
        
        # 连续音频采集
        from conf import AUDIO_ENABLE_CONTINUOUS_COLLECTION, AUDIO_CONTINUOUS_DURATION
        self.continuous_audio_start_time = None  # 连续音频开始时间
        self.continuous_audio_buffer = []  # 连续音频缓冲区
        self.continuous_audio_duration = AUDIO_CONTINUOUS_DURATION  # 连续音频时长（秒）
        self.is_collecting_continuous_audio = False  # 是否正在收集连续音频
        self.enable_continuous_collection = AUDIO_ENABLE_CONTINUOUS_COLLECTION  # 是否启用连续采集
        
        # 配置QoS
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,  # 或者 QoSReliabilityPolicy.RELIABLE
            history=QoSHistoryPolicy.KEEP_LAST,  # 或者 QoSHistoryPolicy.KEEP_ALL
            depth=10  # 根据需要调整队列大小
        )
        
        # 订阅音频话题
        # self.sub_origin_audio = self.node.create_subscription(
        #     ByteMultiArray,
        #     '/voice/origin_audio',
        #     self.origin_audio_callback,
        #     qos_profile
        # )
        
        self.sub_bf_audio = self.node.create_subscription(
            ByteMultiArray,
            '/voice/bf_audio',
            self.bf_audio_callback,
            qos_profile
        )
        
        self.node.get_logger().info('音频数据收集器已初始化')

    def bf_audio_callback(self, msg):
        try:
            data_as_bytes = b''.join(msg.data)
            current_time = time.time()
            
            # 计算音频数据长度（字节数）
            audio_length_bytes = len(data_as_bytes)
            # 计算音频时长（秒）- 假设16位采样，单声道
            audio_duration_seconds = audio_length_bytes / (16000 * 2)  # 16000Hz * 2字节/样本
            
            # 连续音频采集
            if self.enable_continuous_collection and self.is_collecting_continuous_audio:
                if self.continuous_audio_start_time is None:
                    self.continuous_audio_start_time = current_time
                
                elapsed_time = current_time - self.continuous_audio_start_time
                if elapsed_time < self.continuous_audio_duration:
                    # 还在收集时间内，添加到连续音频缓冲区
                    self.continuous_audio_buffer.append((current_time, data_as_bytes))
                    if len(self.continuous_audio_buffer) % 50 == 0:  # 每50帧打印一次
                        print(f"连续音频采集中 - 已采集: {elapsed_time:.2f}秒, 音频帧数: {len(self.continuous_audio_buffer)}, 当前帧长度: {audio_length_bytes}字节")
                elif elapsed_time >= self.continuous_audio_duration:
                    # 收集时间结束，但继续收集直到达到目标帧数
                    expected_frames = int(self.continuous_audio_duration * 16000 * 2 / audio_length_bytes)  # 计算期望的帧数
                    if len(self.continuous_audio_buffer) < expected_frames:
                        # 如果帧数不够，继续收集
                        self.continuous_audio_buffer.append((current_time, data_as_bytes))
                        if len(self.continuous_audio_buffer) % 50 == 0:
                            print(f"连续音频补充中 - 已采集: {elapsed_time:.2f}秒, 音频帧数: {len(self.continuous_audio_buffer)}/{expected_frames}")
                    else:
                        # 达到目标帧数，停止收集
                        self.is_collecting_continuous_audio = False
                        print(f"连续音频采集完成，总时长: {elapsed_time:.2f}秒，音频帧数: {len(self.continuous_audio_buffer)}/{expected_frames}")
            
            # 保持队列大小，移除旧数据
            if len(self.bf_audio_buffer) > 200:  # 增加缓冲区大小
                self.bf_audio_buffer.pop(0)
                
            if not self.bf_audio_queue.full():
                self.bf_audio_queue.put(data_as_bytes)
        except Exception as e:
            self.node.get_logger().error(f"波束形成音频数据处理错误: {str(e)}")

    def start_continuous_audio_collection(self):
        """启动连续音频采集"""
        if not self.enable_continuous_collection:
            print("连续音频采集已禁用")
            return
            
        self.is_collecting_continuous_audio = True
        self.continuous_audio_start_time = None
        self.continuous_audio_buffer.clear()
        print(f"开始连续音频采集，目标时长: {self.continuous_audio_duration}秒")

    def get_continuous_audio_data(self):
        """获取连续音频数据"""
        if not self.continuous_audio_buffer:
            return None
        
        # 合并所有音频数据
        all_audio_bytes = b''
        total_frames = len(self.continuous_audio_buffer)
        
        for i, (_, audio_bytes) in enumerate(self.continuous_audio_buffer):
            all_audio_bytes += audio_bytes
        
        # 计算实际音频时长
        actual_duration = len(all_audio_bytes) / (16000 * 2)  # 16位采样，单声道
        #print(f"连续音频数据统计 - 总帧数: {total_frames}, 总字节数: {len(all_audio_bytes)}, 实际时长: {actual_duration:.2f}秒")
        
        return all_audio_bytes

    def get_latest_audio_frames(self):
        """获取最新的音频帧数据"""
        bf_audio = None
        
        # 获取波束形成音频
        try:
            if not self.bf_audio_queue.empty():
                bf_audio = self.bf_audio_queue.get_nowait()
        except queue.Empty:
            pass
            
        return None, bf_audio


class saveInfo():
    def __init__(self) -> None:
        self.qpos = []
        self.image_list = [None, None, None, None, None, None, None]
        self.image_head_list = [None, None, None]
        self.cmd_vel = [0, 0, 0, 0, 0, 0]
        self.squat_des = [0, 0.92]
        timestampes = int(time.time()*1e9)
        self.image_timestamps_list = [timestampes, timestampes, timestampes]
        self.task_mode = 0


class dataFlag():
    def __init__(self) -> None:
        self.camera_flag = False
        self.head_camera_flag = False
        self.hand_l_camera_flag = False
        self.hand_r_camera_flag = False
        self.qpos_flag = False
        self.cmd_vel_flag = False
        self.squat_des = False
        self.bad_data = False


class saveSate():
    def __init__(self) -> None:
        self.ready_to_save = False
        self.start_get_data = False
        self.saving = False


qos_profile_mocap = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10
)


class MocapSaveRos(Node):
    def __init__(self):
        super().__init__('MocapSaveRos')
        self.subscription_act_qpos = self.create_subscription(Float32MultiArray, '/act_qpos', self.act_qpos_callback, 10)
        self.subscription_cmd_vel = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.subscription_squat_des = self.create_subscription(Float32MultiArray, '/squat_des', self.squat_des_callback, 10)
        self.subscription_save_data = self.create_subscription(String, '/start_save_data', self.start_save_callback, 10)
        self.subscription_task_mode = self.create_subscription(Int32, '/task_mode', self.update_task_mode, 10)
        self.subscription_bad_data = self.create_subscription(Int32, '/bad_data', self.bad_data_callback, 10)
        self.publisher_save_complete = self.create_publisher(Int32, '/save_data_complete', 10)
        self.period_30Hz_m_test = 0
        self.period_30Hz_m_time = time.time()*1000
        self.step = 0
        self.save_state = saveSate()
        self.session_id = None
        self.start_timestamp = None
        self.target_frames = 0
        self.data_info = saveInfo()
        self.data_flag = dataFlag()
        self.running = True
        self.period_update_data_thread = threading.Thread(target=self.period_update_data_m, daemon=True)
        self._remote_img_init()
        self.depth_count = 0
        self.timestamp_pre = time.time()*1000
        if QPOS == "TEST":
            self.data_flag.qpos_flag = True
        self.period_update_data_thread.start()

    def period_update_data_m(self):
        pass

    def _remote_img_init(self):
        self.cv_bridge_head = CvBridge()
        self.cv_bridge_left_hand = CvBridge()
        self.cv_bridge_right_hand = CvBridge()
        self.cv_bridge_head2_hand = CvBridge()
        self.cv_bridge_depth = CvBridge()
        self.subscription = self.create_subscription(SyncImg, "/multi_camera/sync_img", self.image_head_callback, qos_profile=qos_profile_mocap)
        self.sub_check_rgb = self.create_subscription(Image, '/d435/d435/color/image_raw', self.image_check_callback, 10)

    def image_head_callback(self, msg):
        self.data_flag.camera_flag = True
        self.data_flag.head_camera_flag = True
        stream_imagefl = np.frombuffer(msg.imgfl_array, dtype=np.uint8)
        stream_imagef = np.frombuffer(msg.imgf_array, dtype=np.uint8)
        stream_imagefr = np.frombuffer(msg.imgfr_array, dtype=np.uint8)
        self.data_info.image_head_list[0] = stream_imagefl
        self.data_info.image_head_list[1] = stream_imagef
        self.data_info.image_head_list[2] = stream_imagefr
        self.data_info.image_timestamps_list[0] = int(time.time()*1e9)

    def image_check_callback(self, msg):
        cv_image = self.cv_bridge_right_hand.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        size = (640, 480)
        cv_image = cv2.resize(cv_image, size)
        image_show = cv2.resize(cv_image, (1280, 1080))
        cv2.imshow("check", image_show)
        cv2.waitKey(1)

    def act_qpos_callback(self, msg):
        self.data_info.qpos = copy.deepcopy(msg.data)
        self.data_flag.qpos_flag = True

    def cmd_vel_callback(self, msg):
        linear, angular = msg.linear, msg.angular
        self.data_info.cmd_vel = np.array([linear.x, linear.y, linear.z, angular.x, angular.y, angular.z])
        self.data_flag.cmd_vel_flag = True

    def squat_des_callback(self, msg):
        self.data_info.squat_des = np.array([msg.data[0], msg.data[1]])
        self.data_flag.squat_des_flag = True

    def restart_save_data(self, start_timestamp=None, target_frames=None, session_id=None, collection_start_time=None):
        pass

    def save_data(self):
        pass

    def start_save_callback(self, msg):
        command = msg.data
        self.get_logger().info(f'📡 数据采集收到指令: {command}')
        if command.startswith("start:"):
            self.data_flag.bad_data = False
            self.data_info.task_mode = 0
            parts = command.split(":")
            if len(parts) >= 4:
                session_id = parts[1]
                start_timestamp = float(parts[2])
                target_frames = int(parts[3])
                self.session_id = session_id
                self.start_timestamp = start_timestamp
                self.target_frames = target_frames
                self.start_collection_with_timing(session_id, start_timestamp, target_frames)
        elif command == "stop":
            self.save_data()
            print("stop command received")

    def start_collection_with_timing(self, session_id, start_timestamp, target_frames):
        pass

    def update_task_mode(self, msg):
        self.data_info.task_mode = (self.data_info.task_mode + 1) % TASK_NUM_COUNT
        print(f"update task mode : {self.data_info.task_mode}")

    def bad_data_callback(self, msg):
        self.data_flag.bad_data = bool(msg.data) or self.data_flag.bad_data
        print(f"update bad_data: {self.data_flag.bad_data}")


class saveDataMocap(MocapSaveRos):
    def __init__(self) -> None:
        # 获取数据集路径
        self.dataset_path = f"{os.path.expanduser('~')}/data/hf_dataset/{REPO_ID}"
        tmp_info_path, tmp_ros_domain_id = self.generate_camera_info_json(output_dir="/tmp")
        # MocapSaveRos.__init__(self)
        self.lcm_unit = lcmUnit()
        while self.lcm_unit.update_estimator_once == True:
            time.sleep(1)
        self.update_timestamp = time.time()*1000
        self.cnt = 0
        print(f"lcm unit is ok!")
        self.max_step = MAX_STEP
        self.time_start = 0
        self.time_start_max = 5

        self.act_num = 30
        self.qpos_list = []

        self.step = 0

        self.save_state = saveSate()

        self.data_unit = lerobotUnit(repo_id=REPO_ID)
        # print(f"self.data_unit.dataset: {self.data_unit.dataset.episode_buffer["episode_index"]}")
        try:
            if tmp_info_path and os.path.exists(tmp_info_path):
                os.makedirs(self.dataset_path, exist_ok=True)
                dst_path = os.path.join(self.dataset_path, f"info_domain_id_{tmp_ros_domain_id}.json")
                shutil.copy2(tmp_info_path, dst_path)
                print(f"✓ 相机信息已保存到: {dst_path}")
                # 拷贝成功后清理临时文件
                try:
                    os.remove(tmp_info_path)
                    print(f"✓ 已删除临时文件: {tmp_info_path}")
                except Exception as e2:
                    print(f"删除临时JSON失败: {e2}")
        except Exception as e:
            print(f"拷贝相机信息JSON失败: {e}")
        # 最后再启动相机与ROS相关线程/资源
        MocapSaveRos.__init__(self)
        
        # 初始化语音配置客户端
        self.voice_config_client = VoiceConfigClient(self)
        
        # 初始化音频数据收集器
        self.audio_collector = AudioDataCollector(self)
        
        # 音频数据缓存
        self.audio_data_cache = {
            # 'origin_audio': [],
            'bf_audio': []
        }
        # # 获取数据集路径
        # self.dataset_path = f"/home/dreame/data/hf_dataset/{REPO_ID}"
        
        # 获取当前chunk编号
        self.chunk_num = self.get_current_chunk_number()
        
        # 创建chunk文件夹
        self.chunk_folder = os.path.join(self.dataset_path, f"chunk_{self.chunk_num:03d}")
        os.makedirs(self.chunk_folder, exist_ok=True)
        
        # 初始化episode相关变量，但不立即创建文件夹
        self.episode_num = None
        self.episode_folder = None
        self.audio_save_dir = None
        self.sleep_time = 2

        self.session_id = None
        self.start_timestamp = None
        self.target_frames = 0
        self.collection_start_time = None
        self.count_read = 0
        self.ready_read_count = 0
        
        # 配置语音系统
        self.setup_voice_system()
        print(f"audio is starting!")
        print(f"当前chunk: {self.chunk_num}, 等待开始录制...")
        
        # 初始化图片保存线程
        # print("开始初始化图片保存线程...")
        # self.image_save_queue = queue.Queue(maxsize=100)
        # print(f"队列创建成功: {self.image_save_queue}")
        # self.image_save_threads = []
        # self.start_image_save_threads()
        # print("图片保存线程初始化完成")

    def start_collection_with_timing(self, session_id, start_timestamp, target_frames):
        """带时间戳和帧数控制的数据采集启动"""
        import time
        
        self.session_id = session_id
        self.start_timestamp = start_timestamp
        self.target_frames = target_frames
        collection_start_time = start_timestamp        
        delay = start_timestamp - time.time()
        self.get_logger().info(f'准备数据采集，会话: {session_id}')
        self.get_logger().info(f'等待{delay:.1f}秒后开始，目标帧数: {target_frames}')
        
        # 重置采集状态
        self.restart_save_data(start_timestamp, target_frames, session_id, collection_start_time)
        
        # 设置目标帧数
        self.max_step = target_frames
        
    def get_current_chunk_number(self):
        """获取当前chunk编号"""
        try:
            # 检查数据集路径是否存在
            if not os.path.exists(self.dataset_path):
                return 0
            
            # 查找现有的chunk文件夹
            existing_chunks = []
            for item in os.listdir(self.dataset_path):
                if item.startswith("chunk_"):
                    try:
                        chunk_num = int(item.split("_")[1])
                        existing_chunks.append(chunk_num)
                    except:
                        continue
            
            # 返回下一个可用的chunk编号
            if existing_chunks:
                return max(existing_chunks) + 1
            else:
                return 0
        except Exception as e:
            print(f"获取chunk编号失败: {e}")
            return 0

    def generate_camera_info_json(self, output_dir=None):
        """生成相机信息JSON文件
        output_dir: 指定输出目录，默认 /tmp
        返回: (output_file_path, ros_domain_id)
        """
        try:
            print("正在生成相机信息JSON文件...")
            
            # 导入相机信息生成模块
            import sys
            sys.path.append('/home/dreame/test/factory_lerobot')
            
            try:
                from generate_correct_format import get_dev_sn_from_remote, get_ros_domain_id_from_remote, get_robot_parameters_from_remote, get_realsense_info, get_orbbec_info
                
                # 获取所有信息
                dev_sn = get_dev_sn_from_remote()
                ros_domain_id = get_ros_domain_id_from_remote()
                if ros_domain_id is None:
                    print("使用默认ROS_DOMAIN_ID: 0")
                    ros_domain_id = "0"
                
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
                
                # 处理dev_sn
                if isinstance(dev_sn, dict) and "dev_sn" in dev_sn:
                    dev_sn_value = dev_sn["dev_sn"]
                else:
                    dev_sn_value = dev_sn
                
                # 创建完整数据结构
                complete_data = {
                    "dev_sn": dev_sn_value,
                    "ros_domain_id": ros_domain_id,
                    "robot_parameters": robot_parameters,
                    "cameras": cameras
                }
                
                # 选择输出目录（默认 /tmp），避免提前占用数据集根目录
                out_dir = output_dir or "/tmp"
                os.makedirs(out_dir, exist_ok=True)
                # 保存JSON文件
                output_file = os.path.join(out_dir, f"info_domain_id_{ros_domain_id}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(complete_data, f, indent=2, ensure_ascii=False)
                
                print(f"✓ 相机信息生成于: {output_file}")
                return output_file, ros_domain_id
                
            except ImportError as e:
                print(f"✗ 导入相机信息生成模块失败: {e}")
                print("请确保generate_correct_format.py文件存在")
                return None, None
                
        except Exception as e:
            print(f"✗ 生成相机信息失败: {e}")
            return None, None

    def get_current_episode_number(self):
        """获取当前episode编号，基于data文件夹中的episode文件"""
        try:
            # 检查data文件夹是否存在
            data_folder = os.path.join(self.dataset_path, "data", f"chunk-{self.chunk_num:03d}")
            if not os.path.exists(data_folder):
                return 0
            
            # 查找data文件夹中的episode文件
            existing_episodes = []
            for item in os.listdir(data_folder):
                if item.startswith("episode_") and item.endswith(".parquet"):
                    try:
                        episode_num = int(item.split("_")[1].split(".")[0])
                        existing_episodes.append(episode_num)
                    except:
                        continue
            
            # 返回下一个可用的episode编号
            if existing_episodes:
                return max(existing_episodes) + 1
            else:
                return 0
        except Exception as e:
            print(f"获取episode编号失败: {e}")
            return 0

        
    def get_current_episode_number(self):
        """获取当前episode编号，基于data文件夹中的episode文件"""
        try:
            # 检查data文件夹是否存在
            data_folder = os.path.join(self.dataset_path, "data", f"chunk-{self.chunk_num:03d}")
            if not os.path.exists(data_folder):
                return 0
            
            # 查找data文件夹中的episode文件
            existing_episodes = []
            for item in os.listdir(data_folder):
                if item.startswith("episode_") and item.endswith(".parquet"):
                    try:
                        episode_num = int(item.split("_")[1].split(".")[0])
                        existing_episodes.append(episode_num)
                    except:
                        continue
            
            # 返回下一个可用的episode编号
            if existing_episodes:
                return max(existing_episodes) + 1
            else:
                return 0
        except Exception as e:
            print(f"获取episode编号失败: {e}")
            return 0

    def start_image_save_threads(self):
        thread_names = ['hand_l_rgb', 'hand_r_rgb', 'hand_l_depth', 'hand_r_depth']
        for name in thread_names:
            thread = threading.Thread(target=self._image_save_worker, args=(name,), daemon=True)
            thread.start()
            self.image_save_threads.append(thread)
        print("threads is ok!")
        
    def _image_save_worker(self, camera_type):
        while True:
            try:
                task = self.image_save_queue.get(timeout=1.0)
                if task is None:
                    break
                image_data, save_path, timestamp = task

                # 根据文件扩展名选择保存方式
                if save_path.endswith('.npy'):
                    # 保存为numpy数组
                    np.save(save_path, image_data)
                else:
                    # 保存为图片
                    cv2.imwrite(save_path, image_data)
                    
                print(f"thread {camera_type} save done: {os.path.basename(save_path)}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"save image {camera_type} error: {e}")
                import traceback
                traceback.print_exc()
                
    def setup_voice_system(self):
        """设置语音系统配置"""
        try:
            result_code, err_str = self.voice_config_client.send_config(enable_voice=True)
            
            if result_code == Config.Response.SUCCESS:
                self.get_logger().info("语音系统配置成功")
            elif result_code == Config.Response.FAILED:
                self.get_logger().error(f"语音系统配置失败: {err_str}")
            else:
                self.get_logger().warning(f"未知响应码: {result_code}, 错误信息: {err_str}")
        except Exception as e:
            self.get_logger().error(f"语音系统配置异常: {str(e)}")

    def save_audio_as_wav(self, audio_bytes, filename, sample_rate=16000):
        """将音频字节数据保存为WAV文件"""
        if audio_bytes is None:
            return None
        
        try:
            # 计算音频数据长度和时长
            audio_length_bytes = len(audio_bytes)
            audio_duration_seconds = audio_length_bytes / (sample_rate * 2)  # 16位采样，单声道
            
            # 将字节数据转换为numpy数组
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # 创建WAV文件
            filepath = os.path.join(self.audio_save_dir, filename)
            with wave.open(filepath, 'w') as wav_file:
                # 设置WAV文件参数
                wav_file.setnchannels(1)  # 单声道
                wav_file.setsampwidth(2)  # 16位采样
                wav_file.setframerate(sample_rate)  # 采样率
                
                # 写入音频数据
                wav_file.writeframes(audio_array.tobytes())
            

            
            return filepath
        except Exception as e:
            self.get_logger().error(f"保存音频文件失败: {str(e)}")
            return None

    def check_save_file(self):
        # 检查episode文件夹是否已创建
        if self.episode_folder is None:
            print("警告：episode文件夹未创建，跳过图片保存")
            return
            
        # 使用当前episode文件夹
        episode_folder = self.episode_folder
        
        # 创建图片和音频文件夹
        images_folder = os.path.join(episode_folder, "images")
        audio_folder = os.path.join(episode_folder, "audio")
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(audio_folder, exist_ok=True)
        
        # 创建头部多相机子文件夹（仅保留头部摄像头）
        headfl_folder = os.path.join(images_folder, "headfl")
        headf_folder = os.path.join(images_folder, "headf")
        headfr_folder = os.path.join(images_folder, "headfr")
        
        # 创建头部文件夹
        for folder in [headfl_folder, headf_folder, headfr_folder]:
            os.makedirs(folder, exist_ok=True)

    def save_image_list(self, image_head_list, image_list, path, step):
        # 检查episode文件夹是否已创建
        if self.episode_folder is None:
            print("警告：episode文件夹未创建，跳过图片保存")
            return
            
        # 使用当前episode文件夹
        episode_folder = self.episode_folder
        
        # 创建图片和音频文件夹
        images_folder = os.path.join(episode_folder, "images")
        audio_folder = os.path.join(episode_folder, "audio")
        # os.makedirs(images_folder, exist_ok=True)
        # os.makedirs(audio_folder, exist_ok=True)
        
        # 创建头部多相机子文件夹（仅保留头部摄像头）
        headfl_folder = os.path.join(images_folder, "headfl")
        headf_folder = os.path.join(images_folder, "headf")
        headfr_folder = os.path.join(images_folder, "headfr")
        os.makedirs(headfl_folder, exist_ok=True)
        os.makedirs(headf_folder, exist_ok=True)
        os.makedirs(headfr_folder, exist_ok=True)

        try:
            start_time = time.time()*1000
            # 保存头部摄像头图片到对应子文件夹
            path_imagefl = os.path.join(headfl_folder, f"{step}.jpg")
            path_imagef = os.path.join(headf_folder, f"{step}.jpg")
            path_imagefr = os.path.join(headfr_folder, f"{step}.jpg")

            
            with open(path_imagefl, "wb") as f:
                f.write(image_head_list[0])
            with open(path_imagef, "wb") as f:
                f.write(image_head_list[1])
            with open(path_imagefr, "wb") as f:
                f.write(image_head_list[2])
            
            # 相机图像保存已移除
            # print(f"💡 相机图像保存功能已移除，仅保留头部多相机数据")
            # 
            # cv2.imwrite(os.path.join(hand_l_rgb_folder, f"image_{timestamp}.png"), image_list[1])
            # cv2.imwrite(os.path.join(hand_r_rgb_folder, f"image_{timestamp}.png"), image_list[2])
            # cv2.imwrite(os.path.join(hand_l_depth_folder, f"depth_{timestamp}.png"), image_list[3])
            # cv2.imwrite(os.path.join(hand_r_depth_folder, f"depth_{timestamp}.png"), image_list[4])


            # np.save(os.path.join(hand_l_depth_folder, f"depth_{timestamp}.npy"), image_list[3])
            # np.save(os.path.join(hand_l_depth_folder, f"depth_{timestamp}.npy"), image_list[4])
            # print(f"save image hand depth time use: {time.time()*1000 - start_time}")
            
            '''
            hand_save_tasks = [
                (image_list[1], os.path.join(hand_l_rgb_folder, f"image_{timestamp}.png"), timestamp),
                (image_list[2], os.path.join(hand_r_rgb_folder, f"image_{timestamp}.png"), timestamp),
                (image_list[3], os.path.join(hand_l_depth_folder, f"depth_{timestamp}.npy"), timestamp),
                (image_list[4], os.path.join(hand_r_depth_folder, f"depth_{timestamp}.npy"), timestamp)
            ]
            
            
            for task in hand_save_tasks:
                try:
                    self.image_save_queue.put(task, timeout=0.1)
                except queue.Full:
                    print(f"图片保存队列已满，跳过保存 step: {self.step}")
                    print(f"队列大小: {self.image_save_queue.qsize()}")
            '''
                 
           # print(f"save image hand time use: {time.time()*1000 - start_time}")
            
        except Exception as e:
            print(f"save file failed! Error: {e}")
            import traceback
            traceback.print_exc()
        # save hand

    def save_audio(self, step):
        # 检查episode文件夹是否已创建
        if self.episode_folder is None:
            print("警告：episode文件夹未创建，跳过音频保存")
            return
            
        try:
            # 获取当前视频帧时间戳
            current_timestamp = time.time()
            
            # 获取最新的音频数据
            # origin_audio = None
            bf_audio = None
            
            # 从音频队列获取最新数据
            try:
                if not self.audio_collector.bf_audio_queue.empty():
                    bf_audio = self.audio_collector.bf_audio_queue.get_nowait()
            except queue.Empty:
                pass
            

            
            # 保存音频为WAV文件
            if bf_audio is not None:
                self.save_audio_as_wav(
                    bf_audio, 
                    f"frame_{step}_bf_audio.wav"
                )
                # 更新音频缓存
                self.audio_data_cache['bf_audio'].append(f"frame_{step}_bf_audio.wav")
        except Exception as e:
            print(f"get audio failed!")

    def update_lerobot_frame(self, timestamp_list, current_timestamp):
        # 真实数据
        qpos = torch.from_numpy(np.array(copy.deepcopy(self.data_info.qpos)))
        # 测试数据
        hand_dict = {
            "state":    copy.deepcopy(self.lcm_unit.current_robot_state.q).astype(np.float32),
            "action":   qpos,
            "torque":   copy.deepcopy(self.lcm_unit.current_robot_state.torque).astype(np.float32),
            "current":   copy.deepcopy(self.lcm_unit.current_robot_state.current).astype(np.float32),
            "vel":      copy.deepcopy(self.lcm_unit.current_robot_state.dot_q).astype(np.float32),
        }

        leg_dict = {
            "state_q":   copy.deepcopy(self.lcm_unit.current_robot_leg_state.state_q).astype(np.float32),
            "state_qd":  copy.deepcopy(self.lcm_unit.current_robot_leg_state.state_qd).astype(np.float32),
            "state_tau": copy.deepcopy(self.lcm_unit.current_robot_leg_state.state_tau).astype(np.float32),
            "cmd_q":     copy.deepcopy(self.lcm_unit.current_robot_leg_state.cmd_q).astype(np.float32),
            "cmd_qd":    copy.deepcopy(self.lcm_unit.current_robot_leg_state.cmd_qd).astype(np.float32),
            "cmd_tau":   copy.deepcopy(self.lcm_unit.current_robot_leg_state.cmd_tau).astype(np.float32),
            "cmd_vel":   np.float32(copy.deepcopy(self.data_info.cmd_vel)),
        }

        squat_des_array = np.float32(copy.deepcopy(self.data_info.squat_des))


        img_dict = {}
        dep_dict = {}
        task_mode = torch.from_numpy(np.array([self.data_info.task_mode],dtype=np.float32))
        descrip = "grip motor"
        timestamp_list = self.data_info.image_timestamps_list
        self.data_unit.update_frame(hand_dict, leg_dict, squat_des_array, img_dict, dep_dict, timestamp_list, task_mode=task_mode, descrip=descrip, timestamp=current_timestamp)

    def update_data(self):
        
        current_time = time.time()
        if self.start_timestamp:
            # current_time = time.time()
            if current_time < self.start_timestamp:
                remaining_wait = self.start_timestamp - current_time
                if self.count_read >= 30:
                    self.get_logger().info(f"数据采集等待{remaining_wait:.1f}秒后开始...")
                    self.get_logger().info(f"数据采集开始，目标帧数: {getattr(self, 'target_frames', self.max_step)}")
                    self.count_read = 0
                self.count_read += 1
                self.ready_read_count = 0
                return  # 还没到时间，直接返回
            else:
                # self.get_logger().info(f"数据采集开始，目标帧数: {getattr(self, 'target_frames', self.max_step)}")
                self.collection_start_time = current_time
                if self.ready_read_count < 30:
                    self.ready_read_count += 1
        else:
            return
        if self.ready_read_count == 1:
            self.get_logger().info(f"start sample: {current_time}")

        # 新增：检查是否达到目标帧数
        if self.target_frames > 0:
            if self.step >= self.target_frames:
                self.get_logger().info(f"数据采集达到目标帧数{self.target_frames}，自动停止")
                self.save_state.start_get_data = False
                self.save_state.ready_to_save = True
                return

        # if self.time_start < self.time_start_max:
        #     self.time_start = self.time_start+1
        #     self.cnt = 0
        #     return

        if self.cnt == 0:
            self.update_timestamp = current_time*1000
        self.cnt+=1
        if self.cnt>=600:
            self.cnt=0
            self.get_logger().info(f"update 600fps time use: {current_time*1000 - self.update_timestamp}")
            self.update_timestamp = current_time*1000
        # if  self.data_info.image_head_list[0] and  self.data_info.image_head_list[1] and  self.data_info.image_head_list[2] and  self.data_info.image_list[1] and  self.data_info.image_list[2] and  self.data_info.image_list[3] and  self.data_info.image_list[4]:
        #     return
        # 保存时间戳
        current_timestamp = current_time  # 秒级时间戳
        time_start = current_time*1000
        # # 真实数据
        # qpos = torch.from_numpy(np.array(copy.copy(self.data_info.qpos)))
        # # 测试数据
        # hand_dict = {
        #     "state":    copy.copy(self.lcm_unit.current_robot_state.q).astype(np.float32),
        #     "action":   qpos,
        #     "torque":   copy.copy(self.lcm_unit.current_robot_state.current_or_torque).astype(np.float32),
        #     "vel":      copy.copy(self.lcm_unit.current_robot_state.dot_q).astype(np.float32),
        # }

        # leg_dict = {
        #     "state_q":   copy.copy(self.lcm_unit.current_robot_leg_state.state_q).astype(np.float32),
        #     "state_qd":  copy.copy(self.lcm_unit.current_robot_leg_state.state_qd).astype(np.float32),
        #     "state_tau": copy.copy(self.lcm_unit.current_robot_leg_state.state_tau).astype(np.float32),
        #     "cmd_q":     copy.copy(self.lcm_unit.current_robot_leg_state.cmd_q).astype(np.float32),
        #     "cmd_qd":    copy.copy(self.lcm_unit.current_robot_leg_state.cmd_qd).astype(np.float32),
        #     "cmd_tau":   copy.copy(self.lcm_unit.current_robot_leg_state.cmd_tau).astype(np.float32),
        # }
        
        # img_dict = {
        #     # "headf": torch.from_numpy(self.data_info.image_list[0]), 
        #     "headfl": torch.from_numpy(self.data_info.image_head_list[0]), 
        #     "headf": torch.from_numpy(self.data_info.image_head_list[1]), 
        #     "headfr": torch.from_numpy(self.data_info.image_head_list[2]), 

        #     "left_hand_rgb": torch.from_numpy(self.data_info.image_list[1]), 
        #     "right_hand_rgb": torch.from_numpy(self.data_info.image_list[2]),
        #     # "left_hand_depth": torch.from_numpy(self.data_info.image_list[3]).unsqueeze(0).repeat(3, 1, 1),
        #     # "right_hand_depth": torch.from_numpy(self.data_info.image_list[4]).unsqueeze(0).repeat(3, 1, 1)
        #     "left_hand_depth": torch.from_numpy(self.data_info.image_list[3]),
        #     "right_hand_depth": torch.from_numpy(self.data_info.image_list[4]),
        #     }
        img_dict = {}
        # dep_dict = {
        #     "left_hand_depth": torch.from_numpy(self.data_info.image_list[3]),
        #     "right_hand_depth": torch.from_numpy(self.data_info.image_list[4]),
        # }
        dep_dict = {
        }
        
        # task_mode = torch.from_numpy(np.array([self.data_info.task_mode],dtype=np.float32))
        descrip = "grip motor"
        # self.save_image_list(self.data_info.image_head_list, self.data_info.image_list, "extracted_images", current_timestamp)
        step_str = f"{self.step:04d}"

        current_timestamp_str = f"{current_timestamp}"


        threading_time_start = current_time*1000
        thread = threading.Thread(target=self.save_image_list, args=(self.data_info.image_head_list, self.data_info.image_list, "extracted_images", step_str))
        thread.start()
        threading_time_use = threading_time_start - current_time*1000

        # 保存音频
        # self.save_audio(self.step)
        # self.save_audio(current_timestamp)
        
        # 更新lerobot数据
        # self.data_unit.update_frame(hand_dict, leg_dict, img_dict, dep_dict, task_mode=task_mode, descrip=descrip)

        timestamp_list = self.data_info.image_timestamps_list
        # timestamp_list = tuple(self.data_info.image_timestamps_list)
        thread_lerobot = threading.Thread(target=self.update_lerobot_frame, args=(timestamp_list, current_timestamp))
        thread_lerobot.start()


        
        # self.data_unit.update_frame(hand_dict, leg_dict, img_dict, dep_dict, timestamp_list, task_mode=task_mode, descrip=descrip, timestamp=current_timestamp)
        
        # print(f"sampling {self.step} with audio data, time use: {time.time()*1000 - time_start}, lerobot timeuse: {time.time()*1000 - lerobot_time}")
        # print(f"sampling {self.step} with audio data, threading timeuse: {threading_time_use}, time use: {time.time()*1000 - time_start}")
        # print(f"sampling {self.step} with audio data, time use: {time.time()*1000 - time_start}")

        # 测试走这个
        # self.save_dict["/observations/qpos"].append(copy.deepcopy(self.lcm_unit.current_robot_state.q))
        # self.save_dict["/observations/qvel"].append(copy.deepcopy(self.lcm_unit.current_robot_state.dot_q))
        # self.save_dict["/observations/qtor"].append(copy.deepcopy(self.lcm_unit.current_robot_state.current_or_torque))
        # self.save_dict["/observations/body_rpy"].append(copy.deepcopy(self.lcm_unit.current_robot_state.body_imu_rpy))
        # self.save_dict["/action"].append(copy.deepcopy(self.data_info.qpos))
        # self.save_dict["/task_mode"].append([self.data_info.task_mode]) 
        # # self.save_dict[f'/observations/images/head_v2'].append(copy.deepcopy(self.data_info.image_list[3]))
        # self.save_dict[f'/observations/images/headf'].append(copy.deepcopy(self.data_info.image_list[0]))
        # self.save_dict[f'/observations/images/handr_depth'].append(copy.deepcopy(self.data_info.image_list[4]))
        # # self.save_dict[f'/observations/images/left_hand'].append(copy.deepcopy(self.data_info.image_list[1]))
        # self.save_dict[f'/observations/images/right_hand'].append(copy.deepcopy(self.data_info.image_list[2]))

        # print(f"task mode : {self.data_info.task_mode}\n")
        self.step +=1 
        
        # 使用target_frames或max_step来控制采集结束
        if self.target_frames > 0:
            target_step = self.target_frames
        else:
            target_step = self.max_step
        
        if(self.step == target_step):
            self.get_logger().info(f"数据采集完成: {self.step}帧")
            self.save_state.start_get_data = False
            self.save_state.ready_to_save = True
        
        # 每30帧报告一次进度
        if self.step % 300 == 0:
            progress = (self.step / self.target_frames) * 100
            remaining_frames = self.target_frames - self.step
            if self.collection_start_time:
                elapsed = current_time - self.collection_start_time
                fps = self.step / elapsed if elapsed > 0 else 0
                self.get_logger().info(f"数据采集: {self.step}/{self.target_frames}帧 ({progress:.1f}%), {fps:.1f}fps, 剩余{remaining_frames}帧")
        

    def save_data(self):
        if(self.save_state.ready_to_save==True):    
            self.save_state.saving = True

            self.data_unit.save_data()
            
            # 保存连续音频数据
            continuous_audio_data = self.audio_collector.get_continuous_audio_data()
            if continuous_audio_data:
                continuous_audio_file = os.path.join(self.audio_save_dir, "continuous_40s_audio.wav")
                self.save_audio_as_wav(continuous_audio_data, "continuous_40s_audio.wav")
                #print(f"连续音频已保存: {continuous_audio_file}")
                print(f"连续音频长度: {len(continuous_audio_data)}字节, 时长: {len(continuous_audio_data)/(16000*2):.2f}秒")
            
            # # 合并音频文件
            # self.get_logger().info("start combine audio files...")
            # origin_file, bf_file = self.merge_all_audio_files()
            
            # # 保存音频数据统计信息
            # self.save_audio_stats()
            self.step = 0
            self.save_state.ready_to_save = False

            self.save_state.saving = False

    def restart_save_data(self, start_timestamp, target_frames, session_id, collection_start_time):
        
        self.save_state.ready_to_save = False
        self.save_state.start_get_data = False
        self.step = 0
        self.time_start = 0
        
        # 🎯 重置时间戳和帧数控制变量
        self.start_timestamp = start_timestamp
        self.target_frames = target_frames
        self.collection_start_time = collection_start_time
        self.session_id = session_id
            
        # time.sleep(1)
        self.data_unit.dataset.clear_episode_buffer()
        
        # 为新的episode创建新的episode文件夹
        self.episode_num = self.get_current_episode_number()
        self.episode_folder = os.path.join(self.chunk_folder, f"episode_{self.episode_num:06d}")
        os.makedirs(self.episode_folder, exist_ok=True)
        
        # 创建音频文件夹
        self.audio_save_dir = os.path.join(self.episode_folder, "audio")
        os.makedirs(self.audio_save_dir, exist_ok=True)
        
        # 清空音频缓存
        self.audio_data_cache = {
            # 'origin_audio': [],
            'bf_audio': []
        }
        
        print(f"开始录制新的episode，chunk: {self.chunk_num}, episode: {self.episode_num}, 保存路径: {self.episode_folder}")
        
        # 启动连续音频采集
        self.audio_collector.start_continuous_audio_collection()
        
        self.save_state.start_get_data = True



    def period_update_data_m(self):
        # print(f"state 1: {self.save_state.start_get_data},{self.save_state.ready_to_save}")
        # cnt = 0
        
        while True:
            try:
                time.sleep(0.001)
                if self.step == 0:
                    time_start = time.time()*1000
                if(self.save_state.start_get_data and (not self.save_state.saving)):
                    # print(f"state 2: {self.data_flag.camera_flag},{self.data_flag.qpos_flag}")
                    if(self.data_flag.camera_flag and self.data_flag.qpos_flag):
                        # time_start = time.time()*1000
                        self.data_flag.camera_flag = False
                        if QPOS != "TEST":
                            # 测试的时候，这个地方不关掉
                            self.data_flag.qpos_flag = False
                        self.update_data()                            
                        # print(f"time use: {time.time()*1000-time_start}")
                
            except Exception as e:
                print(f"period_update_data_m err: {e}")

                
