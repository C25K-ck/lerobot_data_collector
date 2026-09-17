#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gen1机器人LCM适配器
==================

该模块实现了Gen1机器人的LCM通信适配器，支持：
- 上体数据采集（手臂、手部、腰部、头部）
- 腿部数据采集（状态、命令、力矩等）
- 状态估计器数据
- 多线程异步数据处理

集成到通用数据采集框架中，支持lerobot数据集格式。
"""

import logging
import os
import sys
import threading
import time
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# 注入 biped_lcm_types 所在根目录，保证可导入
try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _proj_root = os.path.dirname(_this_dir)
    _proj_parent = os.path.dirname(_proj_root)
    for _p in (
        os.path.join(_proj_parent, "factory_lerobot"),
        os.path.join(_proj_root, "factory_lerobot"),
    ):
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)
            break
except Exception:
    pass

# 尝试导入LCM相关库
try:
    from lcm import LCM
    from biped_lcm_types.python.upper_body_cmd_package import upper_body_cmd_package
    from biped_lcm_types.python.upper_body_data_package import upper_body_data_package
    from biped_lcm_types.python.state_estimator_lcmt import state_estimator_lcmt
    from biped_lcm_types.python.leg_control_data_lcmt import leg_control_data_lcmt
    from biped_lcm_types.python.leg_control_command_lcmt import leg_control_command_lcmt
    LCM_AVAILABLE = True
except ImportError:
    LCM_AVAILABLE = False
    logger.warning("LCM相关库未安装，Gen1 LCM适配器将不可用")


class ArmState:
    """上体状态类，包含手臂、手部、腰部、头部等30维关节数据"""

    def __init__(self):
        self.dimension = 30
        self.default_arm_control_mode = [200 for i in range(14)]
        self.default_hand_control_mode = [200 for dim0 in range(12)]
        self.default_waist_control_mode = [4 for dim0 in range(2)]
        self.default_head_control_mode = [4 for dim0 in range(2)]
        self.default_control_mode = (
            self.default_arm_control_mode +
            self.default_hand_control_mode +
            self.default_waist_control_mode +
            self.default_head_control_mode
        )

        self.is_used = np.zeros(self.dimension, dtype=int)
        self.error_code = np.zeros(self.dimension, dtype=int)
        self.status = np.zeros(self.dimension, dtype=int)
        self.q = np.zeros(self.dimension, dtype=np.float64)
        self.dot_q = np.zeros(self.dimension, dtype=np.float64)
        self.current_or_torque = np.zeros(self.dimension, dtype=np.float64)
        self.torque = np.zeros(self.dimension, dtype=np.float64)
        self.current = np.zeros(self.dimension, dtype=np.float64)

        self.rpy_dimension = 3
        self.body_imu_rpy = np.zeros(self.rpy_dimension, dtype=np.float64)


class LegState:
    """腿部状态类，包含12维关节数据"""

    def __init__(self):
        self.dimension = 12
        self.state_q = np.zeros(self.dimension, dtype=np.float64)
        self.state_qd = np.zeros(self.dimension, dtype=np.float64)
        self.state_tau = np.zeros(self.dimension, dtype=np.float64)

        self.cmd_q = np.zeros(self.dimension, dtype=np.float64)
        self.cmd_qd = np.zeros(self.dimension, dtype=np.float64)
        self.cmd_tau = np.zeros(self.dimension, dtype=np.float64)


class Gen1LCMAdapter:
    """Gen1机器人LCM适配器"""

    def __init__(self, on_update: Callable[[str, np.ndarray], None]) -> None:
        """
        初始化Gen1 LCM适配器

        Args:
            on_update: 数据更新回调函数，参数为(key: str, value: np.ndarray)
        """
        self.on_update = on_update
        self._lcm: Optional[LCM] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # 机器人状态
        self.current_robot_state = ArmState()
        self.current_robot_leg_state = LegState()

        # 更新标志
        self.update_once_arm = False
        self.update_estimator_once = False
        self.update_leg_state_once = False
        self.update_leg_cmd_once = False

        # LCM话题
        self.upper_body_cmd_topic = 'upper_body_cmd'
        self.upper_body_data_topic = 'upper_body_data'
        self.state_estimator_topic = 'state_estimator'
        self.leg_control_data_topic = 'leg_control_data'
        self.leg_control_command_topic = 'leg_control_command'

    def start(self) -> bool:
        """启动LCM适配器"""
        if not LCM_AVAILABLE:
            logger.error("LCM库不可用，无法启动Gen1 LCM适配器")
            return False

        try:
            # 初始化LCM，尝试多个可能的地址
            lcm_addresses = [
                'udpm://239.255.76.67:7667?ttl=1',
                'udpm://239.255.76.67:7667?ttl=1',  # 默认地址
            ]

            for addr in lcm_addresses:
                try:
                    self._lcm = LCM(addr)
                    logger.info(f"Gen1 LCM适配器初始化成功，使用地址: {addr}")
                    break
                except Exception as e:
                    logger.debug(f"尝试LCM地址 {addr} 失败: {e}")
                    continue

            if self._lcm is None:
                logger.error("所有LCM地址都无法连接")
                return False

            # 订阅话题
            self._lcm.subscribe(self.upper_body_data_topic, self._upper_body_data_cb)
            self._lcm.subscribe(self.state_estimator_topic, self._state_estimator_cb)
            self._lcm.subscribe(self.leg_control_data_topic, self._leg_control_data_cb)
            self._lcm.subscribe(self.leg_control_command_topic, self._leg_control_command_cb)

            # 启动监听线程
            self._running = True
            self._thread = threading.Thread(target=self._lcm_loop, daemon=True)
            self._thread.start()

            logger.info("Gen1 LCM适配器已启动")
            return True

        except Exception as e:
            logger.error(f"启动Gen1 LCM适配器失败: {e}")
            return False

    def stop(self) -> None:
        """停止LCM适配器"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._lcm:
            try:
                self._lcm.close()
            except Exception:
                pass
            self._lcm = None
        logger.info("Gen1 LCM适配器已停止")

    def _lcm_loop(self) -> None:
        """LCM消息循环"""
        while self._running and self._lcm:
            try:
                self._lcm.handle_timeout(100)  # 100ms超时
            except Exception as e:
                if self._running:  # 只在运行状态下记录错误
                    logger.debug(f"LCM循环错误: {e}")
                time.sleep(0.01)

    def _upper_body_data_cb(self, channel: str, data: bytes) -> None:
        """上体数据回调"""
        try:
            msg = upper_body_data_package.decode(data)
            self.current_robot_state.q = np.array(msg.curJointPosVec, dtype=np.float32)
            self.current_robot_state.status = np.array(msg.curStatusVec, dtype=np.float32)
            self.current_robot_state.dot_q = np.array(msg.curSpeedVec, dtype=np.float32)
            self.current_robot_state.current = np.array(msg.curCurrentVec, dtype=np.float32)
            self.current_robot_state.torque = np.array(msg.curTorqueVec, dtype=np.float32)

            # 发送数据更新
            self.on_update("hands.state", self.current_robot_state.q)
            self.on_update("hands.vel", self.current_robot_state.dot_q)
            self.on_update("hands.current", self.current_robot_state.current)
            self.on_update("hands.torque", self.current_robot_state.torque)

            self.update_once_arm = True

        except Exception as e:
            logger.debug(f"上体数据解析错误: {e}")

    def _state_estimator_cb(self, channel: str, data: bytes) -> None:
        """状态估计器回调"""
        try:
            msg = state_estimator_lcmt.decode(data)
            self.current_robot_state.body_imu_rpy = np.array(msg.rpy, dtype=np.float32)
            # 某些实现中可能需要将yaw设为0
            self.current_robot_state.body_imu_rpy[2] = 0
            self.update_estimator_once = True
        except Exception as e:
            logger.debug(f"状态估计器数据解析错误: {e}")

    def _leg_control_data_cb(self, channel: str, data: bytes) -> None:
        """腿部控制数据回调"""
        try:
            msg = leg_control_data_lcmt.decode(data)
            self.current_robot_leg_state.state_q = np.array(msg.q, dtype=np.float32)
            self.current_robot_leg_state.state_qd = np.array(msg.qd, dtype=np.float32)
            self.current_robot_leg_state.state_tau = np.array(msg.tau_est, dtype=np.float32)

            # 发送数据更新
            self.on_update("legs.state_q", self.current_robot_leg_state.state_q)
            self.on_update("legs.state_qd", self.current_robot_leg_state.state_qd)
            self.on_update("legs.state_tau", self.current_robot_leg_state.state_tau)

            self.update_leg_state_once = True

        except Exception as e:
            logger.debug(f"腿部控制数据解析错误: {e}")

    def _leg_control_command_cb(self, channel: str, data: bytes) -> None:
        """腿部控制命令回调"""
        try:
            msg = leg_control_command_lcmt.decode(data)
            self.current_robot_leg_state.cmd_q = np.array(msg.q_des, dtype=np.float32)
            self.current_robot_leg_state.cmd_qd = np.array(msg.qd_des, dtype=np.float32)
            self.current_robot_leg_state.cmd_tau = np.array(msg.tau_ff, dtype=np.float32)

            # 发送数据更新
            self.on_update("legs.cmd_q", self.current_robot_leg_state.cmd_q)
            self.on_update("legs.cmd_qd", self.current_robot_leg_state.cmd_qd)
            self.on_update("legs.cmd_tau", self.current_robot_leg_state.cmd_tau)

            self.update_leg_cmd_once = True

        except Exception as e:
            logger.debug(f"腿部控制命令解析错误: {e}")

    def send_upper_body_cmd(self, joint_positions: np.ndarray) -> None:
        """发送上体控制命令（可选，用于控制机器人）"""
        if not self._lcm:
            return

        try:
            cmd_msg = upper_body_cmd_package()
            cmd_msg.isUsed = 0
            cmd_msg.control_mode = (
                [4] * 7 +  # 左臂
                [4] * 7 +  # 右臂
                [4] * 12 + # 手部
                [4] * 2 +  # 腰部
                [4] * 2    # 头部
            )
            cmd_msg.jointPosVec = joint_positions.tolist()
            cmd_msg.jointSpeedVec = [0.0] * 30
            cmd_msg.jointCurrentVec = [0.0] * 30
            cmd_msg.jointTorqueVec = [0.0] * 30
            cmd_msg.jointKp = [40.0] * 30
            cmd_msg.jointKd = [100.0] * 30

            self._lcm.publish(self.upper_body_cmd_topic, cmd_msg.encode())

        except Exception as e:
            logger.debug(f"发送上体控制命令失败: {e}")

    @property
    def is_ready(self) -> bool:
        """检查适配器是否准备就绪"""
        return (
            self.update_once_arm and
            self.update_estimator_once and
            self.update_leg_state_once and
            self.update_leg_cmd_once
        )
