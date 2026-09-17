#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pico遥控器适配器
================

该模块实现了Pico VR遥控器的集成，支持通过Pico控制器按钮控制数据采集：
- A/X按钮: 开始/停止录制
- B/Y按钮: 紧急停止
- 摇杆: 控制机器人运动（可选）

核心特性：
    - 实时监听Pico控制器状态
    - 按钮事件转换为采集控制命令
    - 线程安全的回调机制
    - 自动重连和错误处理
"""

import logging
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入Pico相关库
try:
    import grpc
    from concurrent import futures
    # 导入pico_stream相关模块（需要从factory_lerobot复制或引用）
    import sys
    import os

    # 添加robot_kinemic路径
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _proj_root = os.path.dirname(_this_dir)  # lerobot_data_collector4.0
    _robot_kinemic_path = os.path.join(_proj_root, "robot_kinemic", "robot_kinemic")

    if _robot_kinemic_path not in sys.path:
        sys.path.insert(0, _robot_kinemic_path)

    try:
        from pico_stream import PicoxrControllerStreamer
        from kinemic_utils import rot_to_eul
        try:
            from kinemic_utils import eul_to_quat  # type: ignore
        except Exception:
            eul_to_quat = None  # type: ignore

        # 尝试导入运动学相关模块（用于高级功能）
        try:
            import pinocchio
            from kinemic import loadUrdf
            from mocap_unit import mocapUnit, btnCtrlUnit
            KINEMATICS_AVAILABLE = True
            logger.info("运动学模块加载成功")
        except ImportError:
            KINEMATICS_AVAILABLE = False
            logger.warning("运动学模块不可用，仅提供基础姿态转换")

        PICO_AVAILABLE = True
    except ImportError:
        PICO_AVAILABLE = False
        KINEMATICS_AVAILABLE = False
        logger.warning("无法导入Pico相关模块，请确保factory_lerobot路径正确")

except ImportError:
    PICO_AVAILABLE = False
    logger.warning("gRPC或相关依赖未安装，Pico适配器将不可用")


class PicoControllerAdapter:
    """Pico遥控器适配器，用于监听控制器事件并转换为采集控制命令"""

    def __init__(
        self,
        server_ip: str = "192.168.12.110",
        port: int = 12345,
        on_control: Optional[Callable[[str], None]] = None,
        on_motion: Optional[Callable[[dict], None]] = None,
        enable_kinematics: bool = False,
        urdf_left_path: Optional[str] = None,
        urdf_right_path: Optional[str] = None,
        teleop_runtime: Optional[Any] = None,
    ):
        """
        初始化Pico控制器适配器

        Args:
            server_ip: Pico服务器IP地址
            port: Pico服务器端口
            on_control: 控制命令回调函数，参数为命令字符串
            on_motion: 运动数据回调函数，参数为运动数据字典
            enable_kinematics: 是否启用运动学计算
            urdf_left_path: 左臂URDF文件路径
            urdf_right_path: 右臂URDF文件路径
        """
        self.server_ip = server_ip
        self.port = port
        self.on_control = on_control
        self.on_motion = on_motion
        self.enable_kinematics = enable_kinematics
        self.teleop_runtime = teleop_runtime

        self.streamer: Optional["PicoxrControllerStreamer"] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # 按钮状态跟踪（避免重复触发）
        self._last_btn_state = {
            "left": {"btn_one": False, "btn_two": False, "trigger_index": False, "trigger_hand": False},
            "right": {"btn_one": False, "btn_two": False, "trigger_index": False, "trigger_hand": False}
        }

        # 运动数据缓存
        self._motion_data = {
            "left_controller": None,
            "right_controller": None,
            "head": None
        }

        self._motion_lock = threading.Lock()

        # 运动学相关
        self._kinematics_enabled = enable_kinematics and KINEMATICS_AVAILABLE
        self._left_arm_kinematics = None
        self._right_arm_kinematics = None
        self._left_controller_unit = None
        self._right_controller_unit = None

        if self._kinematics_enabled:
            try:
                # 初始化运动学求解器
                if urdf_left_path:
                    self._left_arm_kinematics = loadUrdf(is_left=True)
                    logger.info(f"左臂URDF加载成功: {urdf_left_path}")
                if urdf_right_path:
                    self._right_arm_kinematics = loadUrdf(is_left=False)
                    logger.info(f"右臂URDF加载成功: {urdf_right_path}")

                # 初始化控制器运动单元（用于平滑控制）
                self._left_controller_unit = mocapUnit([[-0.2618, 0.2618], [0, 0.5], [-1.5708, 1.5708]], 3, 0.01)
                self._right_controller_unit = mocapUnit([[-0.2618, 0.2618], [0, 0.5], [-1.5708, 1.5708]], 3, 0.01)

                logger.info("运动学模块初始化完成")
            except Exception as e:
                logger.error(f"运动学模块初始化失败: {e}")
                self._kinematics_enabled = False

    def start(self) -> bool:
        """启动Pico控制器监听"""
        if not PICO_AVAILABLE:
            logger.error("Pico相关库不可用，无法启动Pico控制器适配器")
            return False

        if not self.server_ip:
            logger.error("未设置Pico服务器IP地址")
            return False

        try:
            # 创建streamer（不录制，只监听）
            self.streamer = PicoxrControllerStreamer(
                ip=self.server_ip,
                port=self.port,
                record=False
            )

            # 等待数据流建立
            timeout = 10.0
            start_time = time.time()
            while self.streamer.latest is None and (time.time() - start_time) < timeout:
                time.sleep(0.1)

            if self.streamer.latest is None:
                logger.error("Pico数据流建立超时")
                return False

            # 启动监听线程
            self._running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()

            logger.info(f"Pico控制器适配器已启动，连接到 {self.server_ip}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"启动Pico控制器适配器失败: {e}")
            return False

    def stop(self) -> None:
        """停止Pico控制器监听"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.streamer = None
        logger.info("Pico控制器适配器已停止")

    def _monitor_loop(self) -> None:
        """主监听循环"""
        while self._running and self.streamer:
            try:
                latest_data = self.streamer.get_latest()
                if latest_data:
                    self._process_controller_data(latest_data)
                    self._process_motion_data(latest_data)
                    if self.teleop_runtime is not None:
                        try:
                            self.teleop_runtime.set_latest(latest_data)
                            self.teleop_runtime.update_mocap_inf_step()
                        except Exception as exc:
                            logger.debug("teleop_runtime 更新失败: %s", exc)

                time.sleep(0.02)  # 50Hz更新频率

            except Exception as e:
                logger.debug(f"Pico监听循环错误: {e}")
                time.sleep(0.1)

    def _process_controller_data(self, data: dict) -> None:
        """处理控制器按钮事件"""
        if not self.on_control:
            return

        try:
            # 处理左侧控制器按钮 - 与factory_lerobot原始逻辑完全一致
            if data.get("left_connect", False):
                self._check_button_events(data, "left", "btn_one", "pico_toggle_control")      # 按钮0 - 开始/停止控制模式
                self._check_button_events(data, "left", "btn_two", "pico_toggle_data_send")   # 按钮1 - 切换数据发送
                self._check_button_events(data, "left", "trigger_hand", "pico_change_task_mode")  # 按钮3 - 改变任务模式

            # 处理右侧控制器按钮 - 与factory_lerobot原始逻辑完全一致
            if data.get("right_connect", False):
                # 右A/右B：按原项目 start_save_data(0/1) 的语义（开始/停止采集）
                self._check_button_events(data, "right", "btn_one", "pico_right_a")  # 右A -> start_save_data(0)
                self._check_button_events(data, "right", "btn_two", "pico_right_b")  # 右B -> start_save_data(1)

        except Exception as e:
            logger.debug(f"处理控制器数据错误: {e}")

    def _check_button_events(self, data: dict, side: str, button: str, command: str) -> None:
        """检查按钮事件并触发回调 - 与factory_lerobot原始逻辑保持一致"""
        current_state = data.get(f"btn_{side[0]}", [])  # btn_l 或 btn_r

        # 获取按钮状态索引 - 与原始factory_lerobot代码对应
        state_idx = self._get_button_index(button)

        if len(current_state) <= state_idx:
            return

        is_pressed = bool(current_state[state_idx])
        was_pressed = self._last_btn_state[side][button]

        # 检测按下事件（上升沿）- 与原始代码逻辑一致
        if is_pressed and not was_pressed:
            logger.info(f"Pico控制器事件: {side} {button} (索引{state_idx}) 按下 -> {command}")
            if self.on_control:
                self.on_control(command)

        # 更新状态
        self._last_btn_state[side][button] = is_pressed

    def _get_button_index(self, button: str) -> int:
        """获取按钮在数组中的索引 - 与factory_lerobot原始逻辑一致"""
        button_indices = {
            "btn_one": 0,        # 按钮0 (A/X按钮)
            "btn_two": 1,        # 按钮1 (B/Y按钮)
            "trigger_index": 2,  # 扳机索引
            "trigger_hand": 3,   # 按钮3 (握把/扳机手柄，对应原始代码的btn[3])
        }
        return button_indices.get(button, -1)

    def _process_motion_data(self, data: dict) -> None:
        """处理运动数据"""
        if not self.on_motion:
            return

        try:
            motion_data = {}

            # 处理头部数据
            if "head" in data and len(data["head"]) > 0:
                head_matrix = data["head"][0]
                if hasattr(head_matrix, 'shape') and head_matrix.shape[0] >= 3:
                    pos_eul = np.zeros(6, dtype=float)
                    pos_eul[:3] = head_matrix[:3, 3]
                    try:
                        pos_eul[3:] = rot_to_eul(head_matrix[:3, :3])
                        motion_data["head"] = {"position": pos_eul[:3], "orientation": pos_eul[3:]}

                        # 如果启用运动学，添加头部控制
                        if self._kinematics_enabled:
                            motion_data["head"]["kinematics"] = self._compute_head_kinematics(pos_eul)
                    except Exception:
                        pass

            # 处理左侧控制器
            if data.get("left_connect", False) and "left_controller_matrix" in data:
                matrix = data["left_controller_matrix"]
                pos_eul = np.zeros(6, dtype=float)
                pos_eul[:3] = matrix[:3, 3]
                try:
                    pos_eul[3:] = rot_to_eul(matrix[:3, :3])

                    controller_data = {
                        "position": pos_eul[:3],
                        "orientation": pos_eul[3:],
                        "buttons": data.get("btn_l", []),
                        "thumbstick": data.get("r_axis_l", [0, 0])
                    }

                    # 如果启用运动学，计算逆运动学
                    if self._kinematics_enabled and self._left_arm_kinematics:
                        joint_angles = self._compute_inverse_kinematics(pos_eul, is_left=True)
                        if joint_angles is not None:
                            controller_data["joint_angles"] = joint_angles
                            controller_data["kinematics_valid"] = True
                        else:
                            controller_data["kinematics_valid"] = False

                    motion_data["left_controller"] = controller_data
                except Exception:
                    pass

            # 处理右侧控制器
            if data.get("right_connect", False) and "right_controller_matrix" in data:
                matrix = data["right_controller_matrix"]
                pos_eul = np.zeros(6, dtype=float)
                pos_eul[:3] = matrix[:3, 3]
                try:
                    pos_eul[3:] = rot_to_eul(matrix[:3, :3])

                    controller_data = {
                        "position": pos_eul[:3],
                        "orientation": pos_eul[3:],
                        "buttons": data.get("btn_r", []),
                        "thumbstick": data.get("r_axis_r", [0, 0])
                    }

                    # 如果启用运动学，计算逆运动学
                    if self._kinematics_enabled and self._right_arm_kinematics:
                        joint_angles = self._compute_inverse_kinematics(pos_eul, is_left=False)
                        if joint_angles is not None:
                            controller_data["joint_angles"] = joint_angles
                            controller_data["kinematics_valid"] = True
                        else:
                            controller_data["kinematics_valid"] = False

                    motion_data["right_controller"] = controller_data
                except Exception:
                    pass

            # 发送运动数据
            if motion_data and self.on_motion:
                with self._motion_lock:
                    self._motion_data.update(motion_data)
                self.on_motion(motion_data)

        except Exception as e:
            logger.debug(f"处理运动数据错误: {e}")

    def get_latest_motion_data(self) -> dict:
        """获取最新的运动数据"""
        with self._motion_lock:
            return dict(self._motion_data)

    def _compute_head_kinematics(self, head_pose: np.ndarray) -> Optional[dict]:
        """计算头部运动学（可选）"""
        if not self._kinematics_enabled:
            return None

        try:
            # 这里可以添加头部运动学计算逻辑
            # 目前返回基础姿态数据
            quat = None
            if callable(eul_to_quat):
                try:
                    quat = eul_to_quat(head_pose[3:])  # type: ignore[misc]
                except Exception:
                    quat = None
            return {"quaternion": quat, "processed_pose": head_pose}
        except Exception as e:
            logger.debug(f"头部运动学计算错误: {e}")
            return None

    def _compute_inverse_kinematics(self, target_pose: np.ndarray, is_left: bool = True) -> Optional[np.ndarray]:
        """计算逆运动学"""
        if not self._kinematics_enabled:
            return None

        kinematics = self._left_arm_kinematics if is_left else self._right_arm_kinematics
        if not kinematics:
            return None

        try:
            # 使用运动学求解器计算关节角度
            # 这里是简化的调用，实际需要根据loadUrdf类的接口调整
            joint_angles = kinematics.compute_ik(target_pose)
            return joint_angles
        except Exception as e:
            logger.debug(f"逆运动学计算错误: {e}")
            return None

    @property
    def kinematics_enabled(self) -> bool:
        """检查运动学是否启用"""
        return self._kinematics_enabled

    @property
    def is_connected(self) -> bool:
        """检查Pico控制器是否连接"""
        if not self.streamer or not self.streamer.latest:
            return False

        latest = self.streamer.latest
        return latest.get("left_connect", False) or latest.get("right_connect", False)


def create_pico_control_handler(collector: object) -> Callable[[str], None]:
    """
    创建Pico控制命令处理器 - 与factory_lerobot原始逻辑保持一致

    Args:
        collector: 数据采集器实例

    Returns:
        处理Pico命令的回调函数
    """

    # 控制状态
    is_control_active = False
    is_data_send_active = False
    current_task_mode = 0

    def handle_pico_command(command: str) -> None:
        nonlocal is_control_active, is_data_send_active, current_task_mode

        try:
            if command == "pico_toggle_control":
                # 左控制器A按钮: 开始/停止控制模式
                is_control_active = not is_control_active
                if is_control_active:
                    logger.info("Pico控制: 开始控制模式")
                    # 可以在这里初始化控制相关的设置
                else:
                    logger.info("Pico控制: 停止控制模式")
                    is_data_send_active = False  # 停止时也停止数据发送

            elif command == "pico_toggle_data_send":
                # 左控制器B按钮: 切换数据发送模式
                if is_control_active:
                    is_data_send_active = not is_data_send_active
                    if is_data_send_active:
                        logger.info("Pico控制: 开始数据发送")
                    else:
                        logger.info("Pico控制: 停止数据发送")
                else:
                    logger.info("Pico控制: 请先启动控制模式")

            elif command == "pico_change_task_mode":
                # 左控制器扳机: 改变任务模式
                current_task_mode = (current_task_mode + 1) % 2  # 在0和1之间切换
                logger.info(f"Pico控制: 切换到任务模式 {current_task_mode}")

            elif command == "pico_start_save_task_0":
                # 右控制器A按钮: 开始保存数据任务0
                logger.info("Pico控制: 开始保存数据 - 任务0")
                if not collector._session_active:
                    collector.start_session_immediate(target_frames=900)  # 30秒录制
                else:
                    logger.info("Pico控制: 录制已在进行中")

            elif command == "pico_start_save_task_1":
                # 右控制器B按钮: 开始保存数据任务1
                logger.info("Pico控制: 开始保存数据 - 任务1")
                if not collector._session_active:
                    collector.start_session_immediate(target_frames=1800)  # 60秒录制
                else:
                    logger.info("Pico控制: 录制已在进行中")

            elif command == "pico_emergency_stop":
                # 紧急停止（不保存）
                logger.warning("Pico控制: 紧急停止")
                collector._session_active = False
                collector._target_frames = None
                collector._scheduled_start_time = None
                is_control_active = False
                is_data_send_active = False

        except Exception as e:
            logger.error(f"Pico命令处理错误: {e}")

    return handle_pico_command


# 示例用法
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def demo_control_handler(command: str):
        print(f"Pico命令: {command}")

    def demo_motion_handler(motion_data: dict):
        print(f"Pico运动数据: {motion_data}")

    adapter = PicoControllerAdapter(
        server_ip="192.168.12.110",
        on_control=demo_control_handler,
        on_motion=demo_motion_handler
    )

    if adapter.start():
        print("Pico适配器启动成功，按Ctrl+C停止")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            adapter.stop()
    else:
        print("Pico适配器启动失败")
