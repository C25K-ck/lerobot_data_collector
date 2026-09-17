# -*- coding: utf-8 -*-
"""
与 factory_lerobot/act_get_data/lcm_unit.py 行为一致：
- 若存在相邻 factory_lerobot 且已安装 lcm、biped_lcm_types，则动态加载真实 lcmUnit；
- 否则使用占位实现（关节状态为 0），保证模块可导入；真机采集请安装 LCM 依赖。
"""
from __future__ import annotations

import importlib.util
import logging
import os
from typing import Any, Type

import numpy as np

logger = logging.getLogger(__name__)

_lcm_unit_cls: Type[Any] | None = None
_load_error: Exception | None = None

_this_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.dirname(_this_dir)
_proj_parent = os.path.dirname(_proj_root)

_candidate_paths = [
    os.path.join(_proj_parent, "factory_lerobot", "act_get_data", "act_get_data", "lcm_unit.py"),
    os.path.join(os.path.dirname(_proj_parent), "factory_lerobot", "act_get_data", "act_get_data", "lcm_unit.py"),
]

for _path in _candidate_paths:
    if not (_path and os.path.isfile(_path)):
        continue
    try:
        _spec = importlib.util.spec_from_file_location("_lcm_factory_lcm_unit", _path)
        if _spec is None or _spec.loader is None:
            continue
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _lcm_unit_cls = getattr(_mod, "lcmUnit", None)
        if _lcm_unit_cls is not None:
            logger.info("已加载 factory_lerobot LCM 实现：%s", _path)
            break
    except Exception as e:
        _load_error = e
        logger.debug("加载 %s 失败：%s", _path, e)


class _ArmStateStub:
    def __init__(self) -> None:
        self.q = np.zeros(30, dtype=np.float64)
        self.dot_q = np.zeros(30, dtype=np.float64)
        self.torque = np.zeros(30, dtype=np.float64)
        self.current = np.zeros(30, dtype=np.float64)


class _LegStateStub:
    def __init__(self) -> None:
        self.state_q = np.zeros(12, dtype=np.float64)
        self.state_qd = np.zeros(12, dtype=np.float64)
        self.state_tau = np.zeros(12, dtype=np.float64)
        self.cmd_q = np.zeros(12, dtype=np.float64)
        self.cmd_qd = np.zeros(12, dtype=np.float64)
        self.cmd_tau = np.zeros(12, dtype=np.float64)


class _LcmUnitStub:
    """无 biped_lcm_types 时的占位：update_estimator_once 恒为 True，关节为 0。"""

    update_estimator_once = True

    def __init__(self) -> None:
        self.current_robot_state = _ArmStateStub()
        self.current_robot_leg_state = _LegStateStub()
        logger.warning(
            "LCM 使用占位实现（关节数据为 0）。真机采集请安装 lcm、biped_lcm_types 并确保可加载 factory_lerobot lcm_unit。"
            + (f" 上次加载错误: {_load_error}" if _load_error else "")
        )


if _lcm_unit_cls is not None:
    lcmUnit = _lcm_unit_cls
else:
    lcmUnit = _LcmUnitStub
