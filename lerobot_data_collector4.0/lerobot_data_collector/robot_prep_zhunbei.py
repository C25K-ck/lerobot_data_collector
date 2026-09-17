# -*- coding: utf-8 -*-
"""与 factory_lerobot/act_p2p/act_p2p/zhunbei_zhua2.py 一致的准备轨迹（单次执行）。"""

from __future__ import annotations

import copy
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _ensure_rk_path() -> None:
    base = Path(__file__).resolve().parent.parent
    rk = base / "robot_kinemic" / "robot_kinemic"
    s = str(rk)
    if s not in sys.path:
        sys.path.insert(0, s)

    # lcm_unit_gripper 依赖 biped_lcm_types，当前仓库在同级 factory_lerobot 下
    factory_root = base.parent / "factory_lerobot"
    if factory_root.exists():
        fs = str(factory_root)
        if fs not in sys.path:
            sys.path.insert(0, fs)


class _P2PMotion:
    def __init__(self) -> None:
        from lcm_unit_gripper import lcmUnit  # type: ignore

        self.lcm_unit = lcmUnit()
        self.speed = 45 / 180 * math.pi
        self.data_list_t = None
        self.d_len = 0
        self.q = [0 for _ in range(30)]

    def get_current_pos(self) -> bool:
        if self.lcm_unit.update_once_arm is True:
            self.q = self.lcm_unit.current_robot_state.q
            return True
        return False

    def get_pos_list(self, start, end):
        from robot.seven_planner_humanoid import get_pos_list_seven_segment  # type: ignore

        self.data_list_t, self.d_len = get_pos_list_seven_segment(start, end, self.speed)
        return self.data_list_t, self.d_len

    def reset_pose(self) -> None:
        for i in range(self.d_len):
            self.lcm_unit.send_to_robot(np.array(self.data_list_t[i]))
            time.sleep(0.004)


def _act_p2p(handler: _P2PMotion, start, end) -> None:
    handler.get_pos_list(start, end)
    handler.reset_pose()


def run_zhunbei_prep_once(wait_lcm_timeout_sec: float = 120.0) -> None:
    """
    执行 zhunbei_zhua2.py 中 main() 的一次完整准备序列。
    阻塞直到完成或超时/异常。
    """
    _ensure_rk_path()
    p2p_motion = _P2PMotion()
    t0 = time.time()
    while time.time() - t0 < wait_lcm_timeout_sec:
        if p2p_motion.get_current_pos():
            break
        time.sleep(0.1)
    else:
        raise TimeoutError("等待 LCM 上肢状态超时（请确认机器人与 lcm_unit_gripper 正常）")

    logger.info("zhunbei: go ready")
    res = p2p_motion.get_current_pos()
    if not res:
        raise RuntimeError("无法读取当前关节位置")
    tt = copy.copy(p2p_motion.q)
    tt[1] = 0.3
    tt[8] = -0.3
    tt[5] = 0.42
    tt[12] = -0.42
    p2p_motion.q[14] = 80
    p2p_motion.q[20] = 80
    tt[14] = 80
    tt[20] = 80
    _act_p2p(p2p_motion, p2p_motion.q, tt)

    tt_pre = copy.copy(tt)
    res = p2p_motion.get_current_pos()
    tt = copy.copy(tt_pre)
    tt[0] = 0.2
    tt[7] = 0.2
    tt[2] = 1.5707963
    tt[4] = -1.5707963
    tt[9] = -1.5707963
    tt[11] = 1.5707963
    tt[6] = 0
    _act_p2p(p2p_motion, tt_pre, tt)

    tt_pre = copy.copy(tt)
    res = p2p_motion.get_current_pos()
    tt = copy.copy(tt_pre)
    tt[0] = 1
    tt[7] = 1
    tt[1] = 0.5
    tt[8] = -0.5
    tt[3] = -1.7
    tt[10] = 1.7
    tt[12] = 0
    tt[26] = 0.0
    _act_p2p(p2p_motion, tt_pre, tt)

    tt_pre = copy.copy(tt)
    res = p2p_motion.get_current_pos()
    tt = copy.copy(tt_pre)
    tt[0] = 0
    tt[7] = 0
    tt[3] = -1.5
    tt[10] = 1.5
    tt[12] = 0
    tt[29] = 0.34
    tt[26] = 0
    tt[5] = 0.42
    tt[12] = -0.42
    _act_p2p(p2p_motion, tt_pre, tt)

    logger.info("zhunbei 准备动作序列完成")
