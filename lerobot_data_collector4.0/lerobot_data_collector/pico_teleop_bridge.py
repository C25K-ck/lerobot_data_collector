# -*- coding: utf-8 -*-
"""Pico 遥操作与数采共用同一路 Pico 流：从 streamer 字典驱动 mocap + IK + LCM（对齐 act_mocap）。"""

from __future__ import annotations

import copy
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
_TELEOP_BRIDGE_AVAILABLE = False


def _robot_kinemic_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "robot_kinemic" / "robot_kinemic"


def _ensure_rk_path() -> Path:
    d = _robot_kinemic_dir()
    s = str(d)
    if s not in sys.path:
        sys.path.insert(0, s)
    # lcm_unit_gripper 里会 import biped_lcm_types，优先注入工作区同级 factory_lerobot 根目录
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "factory_lerobot",
        Path(__file__).resolve().parent.parent / "factory_lerobot",
    ]
    for factory_root in candidates:
        fs = str(factory_root)
        if factory_root.exists() and fs not in sys.path:
            sys.path.insert(0, fs)
            break
    return d


def _interpolate_v3(
    pre_qpos: List[float], target_qpos: List[float], pre_vel: List[float], ratio: float
) -> Tuple[List[float], List[float]]:
    vel = [0.0 for _ in range(len(target_qpos))]
    for i in range(len(target_qpos)):
        vel[i] = (target_qpos[i] - pre_qpos[i]) * ratio + pre_vel[i] * (1.0 - ratio)
    step_qpos = [pre_qpos[i] + vel[i] for i in range(len(target_qpos))]
    return step_qpos, vel


def build_mocap_map_from_dict(info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """与 teleop_pico.TeleopData.update_mocap_info 相同结构。"""
    try:
        from kinemic_utils import rot_to_eul
    except Exception as e:
        logger.debug("kinemic_utils 不可用: %s", e)
        return None

    mocap_map: Dict[str, Any] = {
        "l_h": {"p_e": [0, 0, 0, 0, 0, 0], "btn": [0, 0, 0, 0], "r_axis": [0, 0]},
        "r_h": {"p_e": [0, 0, 0, 0, 0, 0], "btn": [0, 0, 0, 0], "r_axis": [0, 0]},
        "hmd": {"p_e": [0, 0, 0, 0, 0, 0]},
    }

    if info.get("left_connect") and "left_controller_matrix" in info:
        wrist = info["left_controller_matrix"]
        pos_eul = np.zeros(6, dtype=float)
        pos_eul[:3] = wrist[:3, 3]
        pos_eul[3:] = rot_to_eul(wrist[:3, :3])
        mocap_map["l_h"] = {
            "p_e": pos_eul,
            "btn": list(info.get("btn_l", [])),
            "r_axis": list(info.get("r_axis_l", [0, 0])),
        }

    if info.get("right_connect") and "right_controller_matrix" in info:
        wrist = info["right_controller_matrix"]
        pos_eul = np.zeros(6, dtype=float)
        pos_eul[:3] = wrist[:3, 3]
        pos_eul[3:] = rot_to_eul(wrist[:3, :3])
        mocap_map["r_h"] = {
            "p_e": pos_eul,
            "btn": list(info.get("btn_r", [])),
            "r_axis": list(info.get("r_axis_r", [0, 0])),
        }

    if "head" in info and len(info["head"]) > 0:
        head = info["head"][0]
        pos_eul = np.zeros(6, dtype=float)
        pos_eul[:3] = head[:3, 3]
        try:
            pos_eul[3:] = rot_to_eul(head[:3, :3])
        except Exception:
            pass
        mocap_map["hmd"] = {"p_e": pos_eul}

    return mocap_map


class _MocapManage:
    """与 act_mocap.mocapManage 一致，避免 import act_mocap 拉取 ROS。"""

    def __init__(self) -> None:
        from mocap_unit import btnCtrlUnit, mocapUnit  # type: ignore

        self.perl_m = mocapUnit([[-0.2618, 0.2618], [0, 0.5], [-1.5708, 1.5708]], 3, 0.01)
        self.head_m = mocapUnit([[-0.5236, 0.5236], [-0.3491, 0.1745]], 2, 1)
        self.gripper_l_m = btnCtrlUnit(0.00625, [0, 1], 0)
        self.gripper_r_m = btnCtrlUnit(0.00625, [0, 1], 0)

    def update_head_state(self, head_p_e: np.ndarray) -> None:
        from kinemic_utils import eul_to_quat

        head_quat = eul_to_quat(head_p_e[3:])
        self.head_m.move([head_quat[2], -1 * head_quat[0]])

    def update_perl_state(self, perl_p_e: np.ndarray) -> None:
        from kinemic_utils import eul_to_quat

        perl_quat = eul_to_quat(perl_p_e[3:])
        self.perl_m.move([0, -1 * perl_quat[0], 2 * perl_quat[1]])

    def update_gripper_state(self, btn_l: Any, btn_r: Any) -> None:
        btn_idx = 2
        move_speed = 5
        if btn_l[btn_idx]:
            self.gripper_l_m.move(move_speed)
        else:
            self.gripper_l_m.move(-move_speed)
        if btn_r[btn_idx]:
            self.gripper_r_m.move(move_speed)
        else:
            self.gripper_r_m.move(-move_speed)

    def update_state(self, mocap_data: Any) -> None:
        self.update_head_state(mocap_data.hmd_ctrl.deta_pos_eul)
        self.update_perl_state(mocap_data.chest_ctrl.deta_pos_eul)
        self.update_gripper_state(mocap_data.left_arm_ctrl.btn, mocap_data.right_arm_ctrl.btn)

    def get_perl_state(self) -> List[float]:
        return [self.perl_m.val[0], self.perl_m.val[1], self.perl_m.val[2]]

    def get_head_state(self) -> Any:
        return self.head_m.val

    def get_gripper_state(self) -> Tuple[Any, Any]:
        return self.gripper_l_m.val, self.gripper_r_m.val


def _try_import_runtime_deps() -> bool:
    """IK/mocap 必须可用；LCM 可选（无 biped_lcm_types 时仅不下发真机）。"""
    global _TELEOP_BRIDGE_AVAILABLE
    _ensure_rk_path()
    try:
        from kinemic import loadUrdf  # noqa: F401
        from mocap_data_manage import mocapDataManage  # noqa: F401
        from mocap_unit import mocapUnit  # noqa: F401
        _TELEOP_BRIDGE_AVAILABLE = True
        return True
    except Exception as e:
        logger.warning("Pico 遥操作桥核心依赖未就绪：%s", e)
        _TELEOP_BRIDGE_AVAILABLE = False
        return False


def _try_import_lcm() -> Tuple[Any, Any]:
    _ensure_rk_path()
    from lcm_unit_gripper import lcmUnit  # type: ignore
    from p2p_motion import P2PMotion  # type: ignore

    u = lcmUnit()
    return u, P2PMotion(u)


def teleop_bridge_available() -> bool:
    return _TELEOP_BRIDGE_AVAILABLE or _try_import_runtime_deps()


class PicoTeleopRuntime:
    """单路 Pico 流：update_mocap（随 Pico 线程）+ period_get_data（独立 100Hz 线程）。"""

    def __init__(self) -> None:
        if not _try_import_runtime_deps():
            raise RuntimeError("无法加载 pico_teleop_bridge 依赖，请检查 robot_kinemic")

        from kinemic import loadUrdf  # type: ignore
        from mocap_data_manage import mocapDataManage  # type: ignore

        self._lock = threading.RLock()
        self._latest_info: Optional[Dict[str, Any]] = None
        self.mocap_data_manage = mocapDataManage()
        self.kinemic_manage_l = loadUrdf(True)
        self.kinemic_manage_r = loadUrdf(False)
        self.mocap_m = _MocapManage()
        self.pre_arm: List[float] = [0.0] * 14
        self.pre_arm_vel: List[float] = [0.0] * 14
        self.is_start = False
        self.is_send_data = False
        self.is_real = os.getenv("PICO_TELEOP_IS_REAL", "1").strip().lower() in ("1", "true", "yes", "on")
        self.lcm_unit = None
        self.p2p_m = None
        if self.is_real:
            try:
                self.lcm_unit, self.p2p_m = _try_import_lcm()
            except Exception as e:
                logger.warning("LCM/P2P 不可用，将不下发真机（仅 IK 预览）：%s", e)
                self.is_real = False
        self._running = False
        self._period_thread: Optional[threading.Thread] = None
        self._last_wait_p2p_log_ts = 0.0
        self._lcm_sent_once = False

    def set_latest(self, info: Optional[Dict[str, Any]]) -> None:
        if not info:
            return
        with self._lock:
            self._latest_info = info

    def handle_command(self, cmd: str) -> None:
        """左键 A/B/扳机 等；右 A/B 仍由 Qt _on_pico_control 处理采集。"""
        if cmd == "pico_toggle_control":
            with self._lock:
                self.is_start = not self.is_start
                started = self.is_start
                if not self.is_start:
                    self.is_send_data = False
            if started and self.p2p_m is not None:

                def _p2p() -> None:
                    try:
                        self.p2p_m.start_p2p_move()
                        logger.info("Pico 遥操作：P2P 对齐完成（is_get_q=%s），可按左 B 下发", getattr(self.p2p_m, "is_get_q", False))
                    except Exception as e:
                        logger.warning("start_p2p_move: %s", e)

                threading.Thread(target=_p2p, daemon=True).start()
            logger.info("Pico 遥操作 is_start=%s", self.is_start)
            return
        if cmd == "pico_toggle_data_send":
            with self._lock:
                if not self.is_start:
                    logger.info("请先左 A 进入控制模式")
                    return
                self.is_send_data = not self.is_send_data
                sd = self.is_send_data
            p2p = self.p2p_m
            if sd and p2p is not None:
                if not getattr(p2p, "is_get_q", False):
                    logger.warning(
                        "Pico 左B：已请求下发，但 P2P 尚未对齐（左A 的 start_p2p 未完成或 LCM 未收到上肢状态）。请稍等再按左B。"
                    )
                else:
                    logger.info("Pico 遥操作 is_send_data=%s（P2P 已就绪，将 send_to_robot）", sd)
            else:
                logger.info("Pico 遥操作 is_send_data=%s", sd)
            return
        if cmd == "pico_change_task_mode":
            logger.info("Pico 切换任务模式（未接 ROS publisher，可扩展）")

    def update_mocap_inf_step(self) -> None:
        with self._lock:
            if self._latest_info is None:
                return
            info = self._latest_info
        mocap_map = build_mocap_map_from_dict(info)
        if mocap_map is None:
            return
        with self._lock:
            self.mocap_data_manage.update_button(mocap_map)
            if not self.is_start:
                return
            self.mocap_data_manage.parse_data(mocap_map)
            self.kinemic_manage_l.update_target_pos_diff(self.mocap_data_manage.left_arm_ctrl.deta_pos_eul)
            self.kinemic_manage_r.update_target_pos_diff(self.mocap_data_manage.right_arm_ctrl.deta_pos_eul)

    def period_get_data_step(self) -> None:
        ratio = 0.6
        mocap_data: Optional[List[float]] = None
        is_send = False
        lcm_u = None
        p2p = None
        with self._lock:
            if not self.is_start:
                return
            try:
                next(self.kinemic_manage_l.iteration)
                next(self.kinemic_manage_r.iteration)
            except Exception as e:
                logger.debug("kinemic iteration: %s", e)
                return

            arm_list_q = [0.0] * 14
            qpos_mark_l = [1, 1, 1, 1, 1, 1, 1]
            qpos_mark_r = [1, 1, 1, 1, 1, 1, -1]
            for i in range(7):
                arm_list_q[i] = self.kinemic_manage_l.qpos[i] * qpos_mark_l[i]
                arm_list_q[i + 7] = self.kinemic_manage_r.qpos[i] * qpos_mark_r[i]
            arm_list_q[4] = (arm_list_q[4] + 2.10363551) * 2 - 2.10363551
            arm_list_q[11] = (arm_list_q[11] - 2.10363551) * 2 + 2.10363551

            arm_list, new_vel = _interpolate_v3(self.pre_arm, arm_list_q, self.pre_arm_vel, ratio)
            self.pre_arm = copy.deepcopy(arm_list)
            self.pre_arm_vel = copy.deepcopy(new_vel)

            self.mocap_m.update_state(self.mocap_data_manage)

            mocap_data = [0.0] * 30
            mocap_data[:14] = arm_list
            mocap_data[28:30] = [0.0, 0.34]
            gr = self.mocap_m.get_gripper_state()
            mocap_data[14] = gr[0]
            mocap_data[20] = gr[1]
            is_send = self.is_send_data
            lcm_u = self.lcm_unit
            p2p = self.p2p_m

        if mocap_data is None:
            return
        if self.is_real and is_send and lcm_u is not None and p2p is not None:
            if not getattr(p2p, "is_get_q", False):
                now = time.time()
                if now - self._last_wait_p2p_log_ts > 2.0:
                    self._last_wait_p2p_log_ts = now
                    logger.warning(
                        "Pico 遥操作：is_send_data=True 但 P2P 未对齐，跳过 LCM（避免 p2p_motion.update_qpos 触发 sys.exit）"
                    )
                return
            try:
                lcm_data = copy.deepcopy(mocap_data)
                lcm_data = p2p.update_qpos(lcm_data)
                lcm_u.send_to_robot(np.array(lcm_data))
                if not self._lcm_sent_once:
                    self._lcm_sent_once = True
                    logger.info("Pico 遥操作：首次 LCM send_to_robot 已发出")
            except Exception as e:
                logger.warning("send_to_robot 失败：%s", e)

    def _period_loop(self) -> None:
        while self._running:
            try:
                self.period_get_data_step()
            except Exception as e:
                logger.debug("period_get_data_step: %s", e)
            time.sleep(0.01)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._period_thread = threading.Thread(target=self._period_loop, daemon=True)
        self._period_thread.start()
        logger.info("PicoTeleopRuntime 周期线程已启动 (100Hz)")

    def stop(self) -> None:
        self._running = False
        if self._period_thread and self._period_thread.is_alive():
            self._period_thread.join(timeout=1.0)
        self._period_thread = None
        with self._lock:
            self.is_start = False
            self.is_send_data = False
